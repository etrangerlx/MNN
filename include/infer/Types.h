#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace infer {

// Data type values are kept identical to MNN's schema so that model
// conversion is a plain cast.
enum DataType : int {
    DT_INVALID = 0,
    DT_FLOAT   = 1,
    DT_DOUBLE  = 2,
    DT_INT32   = 3,
    DT_UINT8   = 4,
    DT_INT16   = 5,
    DT_INT8    = 6,
    DT_INT64   = 9,
    DT_BOOL    = 10,
    DT_HALF    = 19,
};

inline size_t dataTypeBytes(DataType t) {
    switch (t) {
        case DT_FLOAT:
        case DT_INT32:
            return 4;
        case DT_DOUBLE:
        case DT_INT64:
            return 8;
        case DT_UINT8:
        case DT_INT8:
        case DT_BOOL:
            return 1;
        case DT_INT16:
        case DT_HALF:
            return 2;
        default:
            return 0;
    }
}

inline const char* dataTypeName(DataType t) {
    switch (t) {
        case DT_FLOAT:
            return "float32";
        case DT_DOUBLE:
            return "float64";
        case DT_INT32:
            return "int32";
        case DT_UINT8:
            return "uint8";
        case DT_INT16:
            return "int16";
        case DT_INT8:
            return "int8";
        case DT_INT64:
            return "int64";
        case DT_BOOL:
            return "bool";
        case DT_HALF:
            return "float16";
        default:
            return "invalid";
    }
}

enum ErrorCode : int {
    NO_ERROR = 0,
    OUT_OF_MEMORY,
    INVALID_INPUT,
    NOT_SUPPORT,
    COMPUTE_ERROR,
};

inline const char* errorCodeName(ErrorCode code) {
    switch (code) {
        case NO_ERROR:
            return "NO_ERROR";
        case OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case INVALID_INPUT:
            return "INVALID_INPUT";
        case NOT_SUPPORT:
            return "NOT_SUPPORT";
        case COMPUTE_ERROR:
            return "COMPUTE_ERROR";
        default:
            return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Operator types. Values are identical to MNN's OpType enumeration.
// ---------------------------------------------------------------------------
enum OpType : int {
    OpType_ArgMax                = 2,
    OpType_BinaryOp              = 7,
    OpType_Cast                  = 9,
    OpType_Concat                = 10,
    OpType_Const                 = 11,
    OpType_Convolution           = 12,
    OpType_ConvolutionDepthwise  = 13,
    OpType_Eltwise               = 22,
    OpType_Flatten               = 28,
    OpType_GatherV2              = 31,
    OpType_InnerProduct          = 33,
    OpType_Input                 = 34,
    OpType_Interp                = 35,
    OpType_MatMul                = 39,
    OpType_Pack                  = 44,
    OpType_Padding               = 45,
    OpType_Permute               = 46,
    OpType_Pooling               = 47,
    OpType_PReLU                 = 49,
    OpType_Reduction             = 68,
    OpType_ReLU                  = 69,
    OpType_ReLU6                 = 70,
    OpType_Reshape               = 73,
    OpType_Scale                 = 77,
    OpType_Shape                 = 80,
    OpType_SliceTf               = 84,
    OpType_Softmax               = 85,
    OpType_Squeeze               = 90,
    OpType_StridedSlice          = 91,
    OpType_Transpose             = 100,
    OpType_UnaryOp               = 101,
    OpType_Unpack                = 102,
    OpType_ConvertTensor         = 129,
    OpType_ArgMin                = 130,
    OpType_Identity              = 262,
};

inline const char* opTypeName(OpType t) {
    switch (t) {
        case OpType_ArgMax:
            return "ArgMax";
        case OpType_BinaryOp:
            return "BinaryOp";
        case OpType_Cast:
            return "Cast";
        case OpType_Concat:
            return "Concat";
        case OpType_Const:
            return "Const";
        case OpType_Convolution:
            return "Convolution";
        case OpType_ConvolutionDepthwise:
            return "ConvolutionDepthwise";
        case OpType_Eltwise:
            return "Eltwise";
        case OpType_Flatten:
            return "Flatten";
        case OpType_GatherV2:
            return "GatherV2";
        case OpType_InnerProduct:
            return "InnerProduct";
        case OpType_Input:
            return "Input";
        case OpType_Interp:
            return "Interp";
        case OpType_MatMul:
            return "MatMul";
        case OpType_Pack:
            return "Pack";
        case OpType_Padding:
            return "Padding";
        case OpType_Permute:
            return "Permute";
        case OpType_Pooling:
            return "Pooling";
        case OpType_PReLU:
            return "PReLU";
        case OpType_Reduction:
            return "Reduction";
        case OpType_ReLU:
            return "ReLU";
        case OpType_ReLU6:
            return "ReLU6";
        case OpType_Reshape:
            return "Reshape";
        case OpType_Scale:
            return "Scale";
        case OpType_Shape:
            return "Shape";
        case OpType_SliceTf:
            return "SliceTf";
        case OpType_Softmax:
            return "Softmax";
        case OpType_Squeeze:
            return "Squeeze";
        case OpType_StridedSlice:
            return "StridedSlice";
        case OpType_Transpose:
            return "Transpose";
        case OpType_UnaryOp:
            return "UnaryOp";
        case OpType_Unpack:
            return "Unpack";
        case OpType_ConvertTensor:
            return "ConvertTensor";
        case OpType_ArgMin:
            return "ArgMin";
        case OpType_Identity:
            return "Identity";
        default:
            return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// Operator parameter enumerations (values identical to MNN).
// ---------------------------------------------------------------------------
enum PadMode : int {
    PadMode_CAFFE = 0,
    PadMode_VALID = 1,
    PadMode_SAME  = 2,
};

enum PoolType : int {
    PoolType_MAXPOOL = 0,
    PoolType_AVEPOOL = 1,
};

enum PoolPadType : int {
    PoolPadType_CAFFE = 0,
    PoolPadType_VALID = 1,
    PoolPadType_SAME  = 2,
};

enum AvgPoolCountType : int {
    AvgPoolCountType_DEFAULT          = 0,
    AvgPoolCountType_INCLUDE_PADDING  = 1,
    AvgPoolCountType_EXCLUDE_PADDING  = 2,
};

enum PadValueMode : int {
    PadValueMode_CONSTANT  = 0,
    PadValueMode_REFLECT   = 1,
    PadValueMode_SYMMETRIC = 2,
};

enum BinaryOpOperation : int {
    BinaryOpOperation_ADD              = 0,
    BinaryOpOperation_SUB              = 1,
    BinaryOpOperation_MUL              = 2,
    BinaryOpOperation_DIV              = 3,
    BinaryOpOperation_MAX_TEMP         = 4,
    BinaryOpOperation_MIN_TEMP         = 5,
    BinaryOpOperation_POW              = 6,
    BinaryOpOperation_REALDIV          = 7,
    BinaryOpOperation_MINIMUM          = 8,
    BinaryOpOperation_MAXIMUM          = 9,
    BinaryOpOperation_GREATER          = 10,
    BinaryOpOperation_GREATER_EQUAL    = 11,
    BinaryOpOperation_LESS             = 12,
    BinaryOpOperation_FLOORDIV         = 13,
    BinaryOpOperation_SquaredDifference = 14,
    BinaryOpOperation_EQUAL            = 15,
    BinaryOpOperation_LESS_EQUAL       = 16,
    BinaryOpOperation_FLOORMOD         = 17,
    BinaryOpOperation_MOD              = 19,
    BinaryOpOperation_ATAN2            = 20,
    BinaryOpOperation_LOGICALOR        = 21,
    BinaryOpOperation_NOTEQUAL         = 22,
    BinaryOpOperation_BITWISE_AND      = 23,
    BinaryOpOperation_BITWISE_OR       = 24,
    BinaryOpOperation_BITWISE_XOR      = 25,
    BinaryOpOperation_LOGICALXOR       = 26,
    BinaryOpOperation_LEFTSHIFT        = 27,
    BinaryOpOperation_RIGHTSHIFT       = 28,
    BinaryOpOperation_MUL_SILU         = 29,
};

enum UnaryOpOperation : int {
    UnaryOpOperation_ABS        = 0,
    UnaryOpOperation_NEG        = 1,
    UnaryOpOperation_FLOOR      = 2,
    UnaryOpOperation_CEIL       = 3,
    UnaryOpOperation_SQUARE     = 4,
    UnaryOpOperation_SQRT       = 5,
    UnaryOpOperation_RSQRT      = 6,
    UnaryOpOperation_EXP        = 7,
    UnaryOpOperation_LOG        = 8,
    UnaryOpOperation_SIN        = 9,
    UnaryOpOperation_COS        = 10,
    UnaryOpOperation_TAN        = 11,
    UnaryOpOperation_ASIN       = 12,
    UnaryOpOperation_ACOS       = 13,
    UnaryOpOperation_ATAN       = 14,
    UnaryOpOperation_RECIPROCAL = 15,
    UnaryOpOperation_LOG1P      = 16,
    UnaryOpOperation_BNLL       = 17,
    UnaryOpOperation_ACOSH      = 18,
    UnaryOpOperation_SINH       = 19,
    UnaryOpOperation_ASINH      = 20,
    UnaryOpOperation_ATANH      = 21,
    UnaryOpOperation_SIGN       = 22,
    UnaryOpOperation_ROUND      = 23,
    UnaryOpOperation_COSH       = 24,
    UnaryOpOperation_ERF        = 25,
    UnaryOpOperation_ERFC       = 26,
    UnaryOpOperation_ERFINV     = 27,
    UnaryOpOperation_EXPM1      = 28,
    UnaryOpOperation_SIGMOID    = 29,
    UnaryOpOperation_TANH       = 30,
    UnaryOpOperation_HARDSWISH  = 31,
    UnaryOpOperation_GELU       = 32,
    UnaryOpOperation_GELU_STANDARD = 33,
    UnaryOpOperation_SILU       = 34,
};

enum ReductionType : int {
    ReductionType_SUM     = 0,
    ReductionType_ASUM    = 1,
    ReductionType_SUMSQ   = 2,
    ReductionType_MEAN    = 3,
    ReductionType_MAXIMUM = 4,
    ReductionType_MINIMUM = 5,
    ReductionType_PROD    = 6,
};

enum EltwiseType : int {
    EltwiseType_PROD    = 0,
    EltwiseType_SUM     = 1,
    EltwiseType_MAXIMUM = 2,
    EltwiseType_SUB     = 3,
};

inline int upDiv(int a, int b) {
    return (a + b - 1) / b;
}

inline int clampInt(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace infer
