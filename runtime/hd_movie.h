// HD movies (hd_movie.c): the player's higher-resolution files in place of the game's full-screen Bink movies.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

struct VkExec;

void hd_movie_romfs_open(const char *rel);                  // romfs.c romfs_open: the RomFS path, relative to the mount
void hd_movie_opened(uintptr_t hbink, int loop, int other);   // frame_rate.c rt_movie_opened: BinkOpen's handle,
                                                              // looping; other: an effect movie, never replaced
void hd_movie_closed(uintptr_t hbink);                      // frame_rate.c rt_movie_close, every movie
void hd_movie_pass_begin(struct VkExec *ex);                // vk_exec.c vk_exec_pass_begin, outside any render pass
extern int hd_movie_live;                                   // a replacement is shown: tex_info asks hd_movie_view
// tex_info (replay thread, texture lock held): the replacement plane for a sampled texture of the game's movie
// planes (format, width, height) at combined slot `slot`, or VK_NULL_HANDLE to keep the game's.
VkImageView hd_movie_view(VkFormat format, uint32_t width, uint32_t height, int slot);
