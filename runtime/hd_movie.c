// HD movies: the player's higher-resolution files in place of the game's full-screen Bink movies.
//
// The event movies (movie/ev%04x.bk2, frame_rate.c) are 1600x900 or 1280x720 Bink 2 files. The game decodes each
// frame on the CPU into three R8 textures, Y at the movie's size and Cr and Cb at half of it, and the interface pass
// draws them into the window texture (at the output's size) with a fragment shader that samples the three planes at
// normalised coordinates; the HUD and menus are drawn after it. So the movie's resolution is only the planes': planes
// of a larger movie, sampled by the same shader, put that movie on the screen at the output's size, under the HUD.
//
// For an event movie with a file movies-hd/<name>.ivf next to the executable (VP9 in IVF, 4:2:0, any size; made by
// the player from the game's movie, e.g. upscaled), this file decodes that file on a thread of its own (libvpx) and,
// while the movie plays, gives tex_info its own Y, U and V images in place of the game's planes. The frame shown is the
// one Bink is at: HBINK's current frame number (+0xC, from 1), read when a pass begins, modulo the file's frames for a
// looping movie. The images take that frame through a host buffer copied before the pass (hd_movie_pass_begin).
//
// Which file: the game opens a movie's file and then BinkOpen on its handle, so the last movie/*.bk2 that romfs_open
// served names the movie BinkOpen returns next. Which textures: R8 at the Bink movie's size (Y) or half of it (chroma)
// while that movie is open; the chroma planes are told apart by their slot (the shader's: Y 0, Cb 1, Cr 2), and
// AC_HDMOVIE_SWAPUV=1 swaps them. AC_HDMOVIE=0 turns the whole feature off. Not done: one effect movie of the same size
// playing on a screen in the scene while an event movie plays would take the event movie's frames too.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vpx/vp8dx.h>
#include <vpx/vpx_decoder.h>

#include "cfg.h"
#include "hd_movie.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define RING 12   // decoded frames kept ahead of the one Bink is at (12 4K frames: 150 MB)

typedef struct {
    uint8_t *y, *u, *v;   // packed planes: w x h, then (w/2) x (h/2) twice
    uint32_t idx;         // frame number from 0 counted on through loops (file frame idx % nframes), UINT32_MAX empty
} Frame;

int hd_movie_live;

static struct {
    PlatMutex mu;
    PlatCond cv;
    int init, off, swap_uv;
    char dir[2048];        // movies-hd folder
    char pending[64];      // the last movie/*.bk2 opened, without the folder and extension; "" none
    // The movie being replaced (game thread sets, replay thread reads)
    uintptr_t hbink;
    uint32_t bw, bh, bframes;
    int loop;              // the player loops the movie: frames go on past the end, from the first again
    // The file and its decoder (decoder thread)
    PlatFile *file;
    const uint8_t *map;
    uint64_t *off_of;      // each frame's data offset in map
    uint32_t *size_of;     // and size
    uint32_t w, h, nframes;
    vpx_codec_ctx_t dec;
    PlatThread *th;
    int stop;
    uint32_t next;         // next frame to decode
    uint32_t want;         // the frame the replay asks for
    Frame ring[RING];
    // Replay thread
    VkImage img[3];
    VmaAllocation mem[3];
    VkImageView view[3];
    uint32_t iw, ih;       // size of the images, 0 none
    int fresh;             // images not yet in GENERAL
    uint32_t shown;        // frame in the images (counted on through loops), UINT32_MAX none
    uint32_t last_cur, loops;   // Bink's frame at the last pass, and the loops counted from its going back
    uint64_t uploads, missed;   // frames uploaded; frames not decoded yet when Bink was at them (each counted once)
    uint32_t missed_at;         // the last frame counted in missed
} M;

static void lazy_init(void) {
    if (M.init) return;
    M.init = 1;
    M.shown = UINT32_MAX;
    for (int i = 0; i < RING; i++) M.ring[i].idx = UINT32_MAX;
    plat_mutex_init(&M.mu);
    plat_cond_init(&M.cv);
    const char *e = cfg_get("AC_HDMOVIE");
    M.off = e && !strcmp(e, "0");
    const char *s = cfg_get("AC_HDMOVIE_SWAPUV");
    M.swap_uv = s && !strcmp(s, "1");
    char exe[2048];
    if (plat_exe_path(exe, sizeof exe)) {
        char *cut = strrchr(exe, '/'), *back = strrchr(exe, '\\');
        if (back > cut) cut = back;
        if (cut) *cut = 0, snprintf(M.dir, sizeof M.dir, "%s/movies-hd", exe);
    }
    if (!M.off && M.dir[0])
        rt_log("hd movie: %s %s", M.dir, plat_dir_exists(M.dir) ? "is there" : "does not exist: the game's movies play");
}

void hd_movie_romfs_open(const char *rel) {
    if (!rel) return;
    const char *m = strstr(rel, "movie/");
    size_t n = strlen(rel);
    if (!m || n < 4 || strcmp(rel + n - 4, ".bk2")) return;
    lazy_init();
    const char *base = strrchr(rel, '/') + 1;
    size_t bn = strlen(base) - 4;
    if (bn >= sizeof M.pending) return;
    plat_mutex_lock(&M.mu);
    memcpy(M.pending, base, bn);
    M.pending[bn] = 0;
    plat_mutex_unlock(&M.mu);
}

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

// IVF: a 32-byte header ("DKIF", version, header size, fourcc, width, height, rate, scale, frames), then per frame a
// 12-byte header (size, 64-bit timestamp) and the data. The frame count is recounted from the frames.
static int ivf_index(void) {
    PlatFileId id;
    if (!plat_file_id(M.file, &id) || id.size < 32) return 0;
    M.map = plat_file_map(M.file);
    if (!M.map || memcmp(M.map, "DKIF", 4) || memcmp(M.map + 8, "VP90", 4)) return 0;
    M.w = rd16(M.map + 12), M.h = rd16(M.map + 14);
    uint32_t cap = 1024, n = 0;
    M.off_of = malloc(cap * sizeof *M.off_of), M.size_of = malloc(cap * sizeof *M.size_of);
    for (uint64_t o = rd16(M.map + 6); o + 12 <= id.size;) {
        uint32_t sz = rd32(M.map + o);
        if (o + 12 + sz > id.size) break;
        if (n == cap) {
            cap *= 2;
            M.off_of = realloc(M.off_of, cap * sizeof *M.off_of), M.size_of = realloc(M.size_of, cap * sizeof *M.size_of);
        }
        M.off_of[n] = o + 12, M.size_of[n++] = sz;
        o += 12 + (uint64_t)sz;
    }
    M.nframes = n;
    return n && M.w >= 16 && M.h >= 16 && !(M.w & 1) && !(M.h & 1);
}

static void free_ring(void) {
    for (int i = 0; i < RING; i++) free(M.ring[i].y), M.ring[i] = (Frame){.idx = UINT32_MAX};
}

// Decoder thread: keeps the frames from `want` on in the ring. Frames are counted on through a looping movie's loops,
// so at the file's end the decoder goes on with its first frame (a key frame) before Bink gets there, and a loop
// starting again finds its first frames decoded.
static void decode_main(void *arg) {
    (void)arg;
    size_t ys = (size_t)M.w * M.h, cs = ys / 4;
    plat_mutex_lock(&M.mu);
    while (!M.stop) {
        uint32_t want = M.want;
        // Asked for a frame behind the ring: decode again from the first frame of that pass through the file, a key
        // frame. Going ahead needs no seek: an inter frame needs the ones before it, so the frames up to `want` are
        // decoded in turn (faster than real time) and only the ring's last RING are kept.
        if (want < M.next && M.ring[want % RING].idx != want) M.next = want - want % M.nframes;
        if ((!M.loop && M.next >= M.nframes) || M.next >= want + RING) {
            plat_cond_wait_ms(&M.cv, &M.mu, 50);
            continue;
        }
        uint32_t v = M.next++, idx = v % M.nframes;
        plat_mutex_unlock(&M.mu);
        vpx_image_t *im = NULL;
        if (vpx_codec_decode(&M.dec, M.map + M.off_of[idx], M.size_of[idx], NULL, 0) == VPX_CODEC_OK) {
            vpx_codec_iter_t it = NULL;
            im = vpx_codec_get_frame(&M.dec, &it);
        }
        plat_mutex_lock(&M.mu);
        if (!im || im->fmt != VPX_IMG_FMT_I420 || im->d_w != M.w || im->d_h != M.h) continue;
        Frame *f = &M.ring[v % RING];
        if (!f->y) f->y = malloc(ys + 2 * cs), f->u = f->y + ys, f->v = f->u + cs;
        for (uint32_t r = 0; r < M.h; r++) memcpy(f->y + (size_t)r * M.w, im->planes[0] + (size_t)r * im->stride[0], M.w);
        for (uint32_t r = 0; r < M.h / 2; r++) {
            memcpy(f->u + (size_t)r * (M.w / 2), im->planes[1] + (size_t)r * im->stride[1], M.w / 2);
            memcpy(f->v + (size_t)r * (M.w / 2), im->planes[2] + (size_t)r * im->stride[2], M.w / 2);
        }
        f->idx = v;
        plat_cond_broadcast(&M.cv);
    }
    plat_mutex_unlock(&M.mu);
}

static void stop_decoder(void) {
    if (!M.th) return;
    plat_mutex_lock(&M.mu);
    M.stop = 1;
    plat_cond_broadcast(&M.cv);
    plat_mutex_unlock(&M.mu);
    plat_thread_join(M.th);
    M.th = NULL;
    vpx_codec_destroy(&M.dec);
    free_ring();
    free(M.off_of), free(M.size_of);
    M.off_of = NULL, M.size_of = NULL;
    if (M.file) plat_file_close(M.file);
    M.file = NULL, M.map = NULL;
}

void hd_movie_opened(uintptr_t h, int loop) {
    lazy_init();
    if (M.off || !h) return;
    char name[64];
    plat_mutex_lock(&M.mu);
    snprintf(name, sizeof name, "%s", M.pending);
    M.pending[0] = 0;
    plat_mutex_unlock(&M.mu);
    if (!name[0] || M.hbink) return;
    const uint32_t *hb = (const uint32_t *)h;
    char path[2200];
    snprintf(path, sizeof path, "%s/%s.ivf", M.dir, name);
    if (!plat_file_exists(path)) return;
    stop_decoder();
    M.file = plat_file_open(path);
    if (!M.file || !ivf_index()) {
        rt_log("hd movie: %s: not a VP9 IVF file of even width and height; the game's movie plays", path);
        stop_decoder();
        return;
    }
    vpx_codec_dec_cfg_t cfg = {.threads = 8, .w = M.w, .h = M.h};
    vpx_codec_err_t err = vpx_codec_dec_init(&M.dec, vpx_codec_vp9_dx(), &cfg, 0);
    if (err != VPX_CODEC_OK) {
        rt_log("hd movie: %s: the VP9 decoder did not start (%s); the game's movie plays", path, vpx_codec_err_to_string(err));
        M.file = (plat_file_close(M.file), NULL);
        free(M.off_of), free(M.size_of);
        M.off_of = NULL, M.size_of = NULL;
        return;
    }
    free_ring();
    M.missed_at = UINT32_MAX;
    M.stop = 0, M.next = 0, M.want = 0, M.shown = UINT32_MAX, M.loop = loop != 0, M.last_cur = 0, M.loops = 0;
    M.bw = hb[0], M.bh = hb[1], M.bframes = hb[2];
    M.th = plat_thread_new(decode_main, NULL, "hd movie");
    __atomic_store_n(&M.hbink, h, __ATOMIC_RELEASE);
    rt_log("hd movie: %s (Bink %ux%u, %u frames) shown from %s.ivf, %ux%u, %u frames%s", name, M.bw, M.bh, M.bframes,
           name, M.w, M.h, M.nframes, M.nframes != M.bframes ? " (not the same count: frames past the end repeat the last)" : "");
}

void hd_movie_closed(uintptr_t h) {
    if (!h || __atomic_load_n(&M.hbink, __ATOMIC_ACQUIRE) != h) return;
    __atomic_store_n(&M.hbink, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&hd_movie_live, 0, __ATOMIC_RELEASE);
    rt_log("hd movie: closed after %llu frames uploaded, %llu of them late (not decoded yet when due)",
           (unsigned long long)M.uploads, (unsigned long long)M.missed);
    M.uploads = M.missed = 0;
    stop_decoder();
}

static int make_images(uint32_t w, uint32_t h) {
    const VkCtx *c = vk_ctx();
    for (int i = 0; i < 3; i++) {
        uint32_t pw = i ? w / 2 : w, ph = i ? h / 2 : h;
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8_UNORM, .extent = {pw, ph, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &M.img[i], &M.mem[i], NULL) != VK_SUCCESS) return 0;
        // The game's planes carry the NVN swizzle 0, 0, 0, R: the shader reads the value from alpha. Every component
        // of these views is the value, whatever the shader reads.
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = M.img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8_UNORM,
                                    .components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                                                   VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R},
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &M.view[i]) != VK_SUCCESS) return 0;
    }
    M.iw = w, M.ih = h, M.fresh = 1;
    return 1;
}

typedef struct { VkBuffer b; VmaAllocation a; } Staging;
static void free_staging(void *arg) {   // completion thread
    Staging *s = arg;
    vmaDestroyBuffer(vk_ctx()->vma, s->b, s->a);
    free(s);
}

void hd_movie_pass_begin(VkExec *ex) {
    uintptr_t h = __atomic_load_n(&M.hbink, __ATOMIC_ACQUIRE);
    if (!h || !ex->cb) return;
    uint32_t cur = ((const uint32_t volatile *)h)[3];   // Bink's current frame, from 1
    uint32_t want = cur ? cur - 1 : 0;
    if (want >= M.nframes) want = M.nframes - 1;   // a file shorter than the Bink movie holds its last frame
    // A looping movie's frame number goes back to 1 at each loop: count the loops, so frames keep counting on.
    if (M.loop && want + M.nframes / 2 < M.last_cur) M.loops++;
    M.last_cur = want;
    want += M.loops * M.nframes;
    if (want == M.shown) return;
    plat_mutex_lock(&M.mu);
    if (M.want != want) M.want = want, plat_cond_broadcast(&M.cv);
    const Frame *f = &M.ring[want % RING];
    int ready = f->idx == want;
    if (!ready) {
        plat_mutex_unlock(&M.mu);
        if (M.missed_at != want) M.missed_at = want, M.missed++;   // shown late: the last frame stays one more present
        return;
    }
    if (M.iw != M.w || M.ih != M.h) {
        if (M.iw) {   // another size: the old images go after the submissions that read them
            vkDeviceWaitIdle(vk_ctx()->device);
            for (int i = 0; i < 3; i++) vkDestroyImageView(vk_ctx()->device, M.view[i], NULL), vmaDestroyImage(vk_ctx()->vma, M.img[i], M.mem[i]);
            M.iw = M.ih = 0;
        }
        if (!make_images(M.w, M.h)) {
            plat_mutex_unlock(&M.mu);
            rt_log("hd movie: could not make the %ux%u planes; the game's movie plays", M.w, M.h);
            __atomic_store_n(&M.hbink, 0, __ATOMIC_RELEASE);
            return;
        }
    }
    size_t ys = (size_t)M.w * M.h, cs = ys / 4, total = ys + 2 * cs;
    Staging *s = calloc(1, sizeof *s);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo mi;
    if (!s || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &s->b, &s->a, &mi) != VK_SUCCESS) {
        plat_mutex_unlock(&M.mu);
        free(s);
        return;
    }
    memcpy(mi.pMappedData, f->y, total);   // y, u, v are one allocation, packed
    plat_mutex_unlock(&M.mu);
    vmaFlushAllocation(vk_ctx()->vma, s->a, 0, VK_WHOLE_SIZE);

    VkImageMemoryBarrier2 ib[3];
    for (int i = 0; i < 3; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                        .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                        .oldLayout = M.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = M.img[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 3, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkDeviceSize offs[3] = {0, ys, ys + cs};
    for (int i = 0; i < 3; i++) {
        VkBufferImageCopy r = {.bufferOffset = offs[i], .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                               .imageExtent = {i ? M.w / 2 : M.w, i ? M.h / 2 : M.h, 1}};
        vkCmdCopyBufferToImage(ex->cb, s->b, M.img[i], VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
    for (int i = 0; i < 3; i++) {
        ib[i].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, ib[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        ib[i].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, ib[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        ib[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    vkCmdPipelineBarrier2(ex->cb, &di);
    vk_completion_add(ex->fence, free_staging, s);
    M.fresh = 0;
    M.shown = want;
    if (++M.uploads % 900 == 0)
        rt_log("hd movie: %llu frames uploaded, %llu of them late (not decoded yet when due)", (unsigned long long)M.uploads,
               (unsigned long long)M.missed);
    if (!__atomic_load_n(&hd_movie_live, __ATOMIC_ACQUIRE)) {
        __atomic_store_n(&hd_movie_live, 1, __ATOMIC_RELEASE);
        rt_log("hd movie: first frame shown (%u)", want);
    }
}

VkImageView hd_movie_view(VkFormat format, uint32_t width, uint32_t height, int slot) {
    if (format != VK_FORMAT_R8_UNORM || !__atomic_load_n(&M.hbink, __ATOMIC_ACQUIRE) || M.shown == UINT32_MAX || !M.iw)
        return VK_NULL_HANDLE;
    if (width == M.bw && height == M.bh) return M.view[0];
    if (width != M.bw / 2 || height != M.bh / 2) return VK_NULL_HANDLE;
    static int logged;
    if (logged < 4) logged++, rt_log("hd movie: chroma plane at slot %d", slot);
    int cr = (!(slot & 1)) != M.swap_uv;   // the shader's slots: Y, Cb, Cr (seen on the title screen's movie)
    return M.view[cr ? 2 : 1];
}
