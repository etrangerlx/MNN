import sys
import MNN
import MNN.cv as cv
import MNN.numpy as np
import MNN.expr as expr


def main():
    # Parse command-line arguments (matching segment.cpp convention)
    if len(sys.argv) < 4:
        print("Usage: python infer.py model.mnn input.jpg output.png")
        print("Using defaults: deeplabv3_257_mv_gpu.mnn ../../demo.jpeg output.png")
        model_path = 'deeplabv3_257_mv_gpu.mnn'
        input_path = '../../demo.jpeg'
        output_path = 'output.png'
    else:
        model_path = sys.argv[1]
        input_path = sys.argv[2]
        output_path = sys.argv[3]

    # Load the MNN segmentation model
    # C++ equivalent: Interpreter::createFromFile(argv[1])
    net = MNN.nn.load_module_from_file(model_path, [], [])

    # Read input image (returns Var with NHWC layout, uint8, BGR format)
    image = cv.imread(input_path)
    if image is None:
        print(f"Error: Can't open {input_path}")
        return

    # Get original dimensions (NHWC: [H, W, C])
    print(f"origin size: {image.shape[1]}, {image.shape[0]}")

    # Preprocessing: resize + normalize (ImageNet normalization in BGR order)
    # This model expects BGR input with ImageNet standardization:
    #   (x - mean) * norm, where mean/std are ImageNet values in BGR channel order
    # segment.cpp's [-1,1] normalization (mean=127.5, norm=0.00785) is WRONG for this model.
    input_size = 257
    image = cv.resize(image, (input_size, input_size),
                      interpolation=cv.INTER_LINEAR,
                      mean=[103.94, 116.78, 123.68],
                      norm=[0.017, 0.017, 0.017])

    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    input_var = np.expand_dims(image, 0)

    # Convert to NC4HW4 format for MNN internal computation
    input_var = expr.convert(input_var, expr.NC4HW4)

    # Run inference
    # C++ equivalent: net->runSession(session)
    output_var = net.forward(input_var)

    # Post-processing (matching segment.cpp's Express logic)
    # C++: _Convert(output, NHWC)
    output_var = expr.convert(output_var, expr.NHWC)

    # Get output dimensions
    shape = output_var.shape
    height = shape[1]
    width = shape[2]
    channel = shape[3]
    print(f"output: w={width}, h={height}, channel={channel}")

    # C++: _Reshape(output, {-1, channel}); auto kv = _TopKV2(output, _Scalar<int>(1))
    # TopKV2 with k=1 is equivalent to argmax along the channel axis
    output_var = expr.reshape(output_var, [-1, channel])
    pred_class = expr.argmax(output_var, axis=-1)  # dtype=int32, shape=[H*W]

    # C++: auto mask = _Select(_Equal(index, _Scalar<int>(humanIndex)),
    #                           _Scalar<int>(255), _Scalar<int>(0))
    #        mask = _Cast<uint8_t>(mask)
    HUMAN_INDEX = 15
    mask = expr.select(
        expr.equal(pred_class, expr.scalar(HUMAN_INDEX)),
        expr.scalar(255),
        expr.scalar(0)
    )
    mask = expr.cast(mask, expr.uint8)

    # Reshape to [height, width] for grayscale image output
    mask = expr.reshape(mask, [height, width])

    # Save output mask
    # C++ equivalent: stbi_write_png(argv[3], width, height, 1, mask->readMap<uint8_t>(), width)
    cv.imwrite(output_path, mask)
    print(f"Segmentation mask saved to {output_path}")


if __name__ == '__main__':
    main()
