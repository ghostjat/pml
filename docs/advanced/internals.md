# Internals

This page describes the internal implementation of the framework.

## FFI bridge

`src/Lib/TensorEngine.php` declares the native C interface exactly once.

### Key native types

```c
typedef struct {
    int ndim;
    int shape[8];
    size_t stride[8];
    size_t total_size;
    size_t byte_size;
    bool owns_data;
    bool is_arena;
    int dtype;
    void* data;
} TensorC;
```

`TensorC` is the PHP-visible struct used by `Pml\Tensor`.

## Dataset internal modes

- **ETL mode**: `Dataset::$dfPtr` holds a C DataFrame pointer.
- **Tensor mode**: `Dataset::$samples` and `Dataset::$labels` hold numeric tensors.

`Dataset::_ensureTensorMode()` lazily converts the dataset and frees the DataFrame pointer.

## Zero-copy view semantics

`Tensor::slice()` and `Tensor::view()` use native C kernels that return a `TensorC*` view. The PHP wrapper stores the parent tensor reference so the underlying buffer remains valid.

## Persistence internals

- `ModelStore::toArray()` serializes PHP state and skips all `Tensor` and `FFI\CData` values.
- `ModelStore::extractTensors()` collects tensor state from `Stateful` objects or via reflection on `Tensor`-typed properties.
- `SafeTensorsIO` writes tensor bytes via native kernels and loads them with `Tensor::fromMmap()`.

## Neural network internals

- `Pml\NeuralNetwork\Sequential` holds layers, a loss, and an optimizer.
- `Dense` stores parameters and gradients as tensors and implements `Stateful`.
- Backpropagation is performed by calling `backward()` on layers in reverse order.
- The training loop in `Trainer` handles batching, validation, early stopping, and LR scheduling.

## GBDT internals

- `GBDTRegressor` and `GBDTClassifier` build histogram-based trees using `Tensor` kernels.
- The native C backend provides functions such as `tensor_gbdt_train_tree()` and `tensor_gbdt_predict_all()`.
- The PHP class assembles and saves tree structure tensors for inference.

## Example internal flow: `Dataset::fromCSV()`

1. Call `tensor_dataset_from_csv()` in the C backend.
2. If it returns a pointer array, wrap the returned sample and label tensors.
3. If it returns `NULL`, fallback to `df_read_csv()` and use ETL mode.

## Example internal flow: model saving

1. `Pipeline::save()` collects transformer config and tensor state.
2. Transformer tensors are stored in `transformers.safetensors`.
3. The estimator is saved either with its own `Persistable::save()` or via `ModelStore::save()`.
4. `config.json` contains metadata, class names, and transformer prefixes.
