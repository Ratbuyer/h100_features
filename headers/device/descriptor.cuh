#include <cstdint>

// taken from https://github.com/KnowingNothing/MatmulTutorial/blob/18366a51005c3b3395449d5eb5da02ec56198b65/examples/atom/single-wgmma-f8.cu#L169

union GmmaDescriptor
{
  __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}
  __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  __device__ constexpr GmmaDescriptor &
  operator=(GmmaDescriptor const &t) noexcept
  {
    desc_ = t.desc_;
    return *this;
  }

  __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept
  {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct
  {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2; // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  __device__ constexpr operator uint64_t() const noexcept { return desc_; }

  // Printer
  //   __device__ friend void print(GmmaDescriptor const& t)
  //   {
  //     #if !defined(__CUDACC_RTC__)
  //     printf("GmmaDescriptor: 0x%016 %lli\n", static_cast<long
  //     long>(t.desc_)); printf("  start_addr :  0x%04x\n",
  //     t.bitfield.start_address_); printf("  leading_off:  0x%04x (%d)\n",
  //     t.bitfield.leading_byte_offset_, t.bitfield.leading_byte_offset_);
  //     printf("  stride_off :  0x%04x (%d)\n", t.bitfield.stride_byte_offset_,
  //     t.bitfield.stride_byte_offset_); printf("  base_offset:  0x%01x\n",
  //     t.bitfield.base_offset_); printf("  layout_type:  0x%01x (%s)\n",
  //     t.bitfield.layout_type_,
  //     to_string(static_cast<GMMA::LayoutType>(t.bitfield.layout_type_)));
  //     #endif
  //   }
};

template <class PointerType>
__device__ GmmaDescriptor make_desc_a(PointerType smem_ptr)
{
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = 0;         // no swizzle
  desc.bitfield.leading_byte_offset_ = 8; // 16 bytes
  desc.bitfield.stride_byte_offset_ = 16; // 8 bytes
  /// base_offset_ is not valid for non-swizzle
  desc.bitfield.base_offset_ = 0;
  return desc;
}

template <class PointerType>
__device__ GmmaDescriptor make_desc_b(PointerType smem_ptr)
{
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = 0;         // no swizzle
  desc.bitfield.leading_byte_offset_ = 8; // 16 bytes
  desc.bitfield.stride_byte_offset_ = 16; // 8 bytes
  // base_offset_ is not valid for non-swizzle
  desc.bitfield.base_offset_ = 0; // (uint_ptr >> 7) & 7;
  return desc;
}
