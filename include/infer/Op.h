#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infer/Tensor.h"
#include "infer/Types.h"

namespace infer {

struct Conv2DCommon {
    int padX        = 0;
    int padY        = 0;
    int kernelX     = 1;
    int kernelY     = 1;
    int strideX     = 1;
    int strideY     = 1;
    int dilateX     = 1;
    int dilateY     = 1;
    int padMode     = PadMode_CAFFE;
    int group       = 1;
    int outputCount = 0;
    int inputCount  = 0;
    bool relu       = false;
    bool relu6      = false;
    std::vector<int> pads;
};

struct PoolCommon {
    int padX      = 0;
    int padY      = 0;
    int kernelX   = 0;
    int kernelY   = 0;
    int strideX   = 1;
    int strideY   = 1;
    bool isGlobal = false;
    bool ceilModel = true;
    int type      = PoolType_MAXPOOL;
    int padType   = PoolPadType_CAFFE;
    int countType = AvgPoolCountType_DEFAULT;
    std::vector<int> pads;
};

// Flattened parameter storage for every supported operator. Only the fields
// relevant to a given op type are populated by the model loader.
struct OpAttr {
    Conv2DCommon conv;
    PoolCommon pool;

    int binaryOp    = BinaryOpOperation_ADD;
    int unaryOp     = UnaryOpOperation_ABS;
    int reductionOp = ReductionType_SUM;
    int eltwiseType = EltwiseType_SUM;

    int axis  = 0;
    int axis2 = 0;

    std::vector<int> dims;          // Reshape target / generic integer list
    std::vector<int> squeezeDims;   // Squeeze axes
    std::vector<int> permuteDims;   // Permute order
    std::vector<int> reductionDims; // Reduction axes
    std::vector<int> pads;          // Padding / Pool pads

    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> slope;
    std::vector<float> scale;
    std::vector<float> offset;
    std::vector<float> coeff;

    bool keepDims     = false;
    bool transposeA   = false;
    bool transposeB   = false;
    bool alignCorners = false;
    bool halfPixelCenters = false;
    bool hasStart     = false;
    bool hasEnd       = false;

    float minValue   = 0.0f;
    float maxValue   = 6.0f;
    float slopeValue = 0.0f;
    float widthScale = 0.0f;
    float heightScale = 0.0f;
    float widthOffset = 0.0f;
    float heightOffset = 0.0f;
    int outputWidth  = 0;
    int outputHeight = 0;

    int topK        = 1;
    int outMaxVal   = 1;
    int resizeType  = 0;
    int padMode     = PadValueMode_CONSTANT;
    int beginMask   = 0;
    int endMask     = 0;
    int ellipsisMask = 0;
    int newAxisMask  = 0;
    int shrinkAxisMask = 0;
    int start        = 0;
    int end          = 0;
    int outputCount  = 0;
    int channels     = 0;

    DataType dataType = DT_FLOAT;
    // Dimension formats of the source and destination of a ConvertTensor op.
    int srcFormat  = 0;
    int dstFormat  = 0;
};

struct OpDesc {
    OpType type = OpType_Identity;
    std::string name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    OpAttr attr;
};

class Backend;
class Execution {
public:
    explicit Execution(Backend* backend) : mBackend(backend) {
    }
    virtual ~Execution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;

    Backend* backend() const {
        return mBackend;
    }

private:
    Backend* mBackend = nullptr;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    // Returns nullptr if the op is not supported by this backend.
    virtual std::shared_ptr<Execution> onCreate(const OpDesc& op, const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) = 0;
};

class ThreadPool;
std::shared_ptr<Backend> createCPUBackend(ThreadPool* pool);

} // namespace infer
