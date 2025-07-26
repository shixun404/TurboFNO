void cutlass::gemm::kernel::Gemm {
  Mma_,
   Epilogue_,
   ThreadblockSwizzle_,
   SplitKSerial
}::operator()(const cutlass::gemm::kernel::Gemm {
  Mma_,
   Epilogue_,
   ThreadblockSwizzle_,
   SplitKSerial
}::Params &,
 cutlass::gemm::kernel::Gemm {
  Mma_,
   Epilogue_,
   ThreadblockSwizzle_,
   SplitKSerial
}::SharedStorage &) [
  with Mma_ = cutlass::gemm::threadblock::MmaPipelined {
  cutlass::gemm::GemmShape {
    64,
     32,
     8
  },
   cutlass::transform::threadblock::PredicatedTileIterator {
    cutlass::MatrixShape {
      64,
       8
    },
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     1,
     cutlass::transform::PitchLinearStripminedThreadMap {
      cutlass::PitchLinearShape {
        8,
         64
      },
       128,
       1
    },
     1,
     false,
     cutlass::layout::NoPermute
  },
   cutlass::transform::threadblock::RegularTileIterator {
    cutlass::MatrixShape {
      64,
       8
    },
     cutlass::complex {
      float
    },
     cutlass::layout::ColumnMajor,
     1,
     cutlass::transform::TransposePitchLinearThreadMapSimt {
      cutlass::transform::PitchLinearStripminedThreadMap {
        cutlass::PitchLinearShape {
          8,
           64
        },
         128,
         1
      }
    },
     8
  },
   cutlass::transform::threadblock::PredicatedTileIterator {
    cutlass::MatrixShape {
      8,
       32
    },
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     0,
     cutlass::transform::PitchLinearStripminedThreadMap {
      cutlass::PitchLinearShape {
        32,
         8
      },
       128,
       1
    },
     1,
     false,
     cutlass::layout::NoPermute
  },
   cutlass::transform::threadblock::RegularTileIterator {
    cutlass::MatrixShape {
      8,
       32
    },
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     0,
     cutlass::transform::PitchLinearStripminedThreadMap {
      cutlass::PitchLinearShape {
        32,
         8
      },
       128,
       1
    },
     8
  },
   cutlass::complex {
    float
  },
   cutlass::layout::RowMajor,
   cutlass::gemm::threadblock::MmaPolicy {
    cutlass::gemm::warp::MmaSimt {
      cutlass::gemm::GemmShape {
        32,
         16,
         8
      },
       cutlass::complex {
        float
      },
       cutlass::layout::ColumnMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::gemm::warp::MmaSimtPolicy {
        cutlass::MatrixShape {
          8,
           4
        },
         cutlass::layout::RowMajorInterleaved {
          1
        },
         cutlass::gemm::GemmShape {
          2,
           2,
           1
        }
      },
       1,
       cutlass::ComplexTransform::kNone,
       cutlass::ComplexTransform::kNone,
       __nv_bool
    },
     cutlass::MatrixShape {
      2,
       0
    },
     cutlass::MatrixShape {
      0,
       0
    },
     1
  },
   cutlass::NumericArrayConverter {
    cutlass::complex {
      float
    },
     cutlass::complex {
      float
    },
     4,
     cutlass::FloatRoundStyle::round_to_nearest,
     cutlass::transform::thread::UnaryTransform::Identity
  },
   cutlass::NumericArrayConverter {
    cutlass::complex {
      float
    },
     cutlass::complex {
      float
    },
     2,
     cutlass::FloatRoundStyle::round_to_nearest,
     cutlass::transform::thread::UnaryTransform::Identity
  },
   __nv_bool
}; 


Epilogue_ = cutlass::epilogue::threadblock::Epilogue {
  cutlass::gemm::GemmShape {
    64,
     32,
     8
  },
   cutlass::gemm::warp::MmaSimt {
    cutlass::gemm::GemmShape {
      32,
       16,
       8
    },
     cutlass::complex {
      float
    },
     cutlass::layout::ColumnMajor,
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     cutlass::gemm::warp::MmaSimtPolicy {
      cutlass::MatrixShape {
        8,
         4
      },
       cutlass::layout::RowMajorInterleaved {
        1
      },
       cutlass::gemm::GemmShape {
        2,
         2,
         1
      }
    },
     1,
     cutlass::ComplexTransform::kNone,
     cutlass::ComplexTransform::kNone,
     __nv_bool
  },
   1,
   cutlass::epilogue::threadblock::PredicatedTileIterator {
    cutlass::epilogue::threadblock::OutputTileOptimalThreadMap {
      cutlass::epilogue::threadblock::OutputTileShape {
        32,
         1,
         8,
         2,
         1
      },
       cutlass::epilogue::threadblock::OutputTileShape {
        1,
         2,
         2,
         1,
         4
      },
       128,
       1,
       64
    },
     cutlass::complex {
      float
    },
     false,
     cutlass::layout::NoPermute,
     false
  },
   cutlass::epilogue::warp::FragmentIteratorSimt {
    cutlass::gemm::GemmShape {
      32,
       16,
       8
    },
     cutlass::gemm::thread::Mma {
      cutlass::gemm::GemmShape {
        4,
         4,
         1
      },
       cutlass::complex {
        float
      },
       cutlass::layout::ColumnMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::arch::OpMultiplyAdd,
       __nv_bool
    },
     cutlass::layout::RowMajor,
     cutlass::gemm::warp::MmaSimtPolicy {
      cutlass::MatrixShape {
        8,
         4
      },
       cutlass::layout::RowMajorInterleaved {
        1
      },
       cutlass::gemm::GemmShape {
        2,
         2,
         1
      }
    }
  },
   cutlass::epilogue::warp::TileIteratorSimt {
    cutlass::gemm::GemmShape {
      32,
       16,
       8
    },
     cutlass::gemm::thread::Mma {
      cutlass::gemm::GemmShape {
        4,
         4,
         1
      },
       cutlass::complex {
        float
      },
       cutlass::layout::ColumnMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::complex {
        float
      },
       cutlass::layout::RowMajor,
       cutlass::arch::OpMultiplyAdd,
       __nv_bool
    },
     cutlass::complex {
      float
    },
     cutlass::layout::RowMajor,
     cutlass::gemm::warp::MmaSimtPolicy {
      cutlass::MatrixShape {
        8,
         4
      },
       cutlass::layout::RowMajorInterleaved {
        1
      },
       cutlass::gemm::GemmShape {
        2,
         2,
         1
      }
    }
  },
   cutlass::epilogue::threadblock::SharedLoadIterator {
    cutlass::epilogue::threadblock::OutputTileOptimalThreadMap {
      cutlass::epilogue::threadblock::OutputTileShape {
        32,
         1,
         8,
         2,
         1
      },
       cutlass::epilogue::threadblock::OutputTileShape {
        1,
         2,
         2,
         1,
         4
      },
       128,
       1,
       64
    }::CompactedThreadMap,
     cutlass::complex {
      float
    },
     8
  },
   cutlass::epilogue::thread::LinearCombination {
    cutlass::complex {
      float
    },
     1,
     cutlass::complex {
      float
    },
     cutlass::complex {
      float
    },
     cutlass::epilogue::thread::ScaleType::Default,
     cutlass::FloatRoundStyle::round_to_nearest,
     cutlass::complex {
      float
    }
  },
   cutlass::MatrixShape {
    0,
     9
  },
   1,
   1
}; 
ThreadblockSwizzle_ = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle {
  1
}; __nv_bool SplitKSerial = false]
