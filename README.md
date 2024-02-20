# Region-based Similarity Search ONNX 

## Download models

First, download models for segmentation and hashing

```
cd models
./get_models.sh
cd ..
```


## Build

Build Docker image to run ONNX models

```
cd docker
docker build -t rbss/onnx:1.0 .
cd ..
```

## Run

Run image with jupyter notebook locally (NVIDIA GPU required)


```
docker run -v $(pwd):/data -p 8888:8888 --runtime=nvidia -it --rm rbss/onnx:1.0
```

Got to `http://localhost:8888/lab`