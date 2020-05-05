# DeepFakeDetection
Team members: [Cai Ruicong](https://github.com/ray-97), [Choo Ze Yuan](https://github.com/Raikonen), [Huang Linhang](https://github.com/3anlang), [Ye Chenchen](https://github.com/IYVVVVV), [Zhang Xinran](https://github.com/Zhang-Xinran), [Zhou Tianyu](https://github.com/MackyMaguire)

## Download Folder Structure:
- /project
    - model python file here
    - ExtractFrames.py here
    - /data
        - /manipulated_sequences
            - /DeepFakeDection (3068)
                - /c23
                    - /videos
                        - .mp4 files
            - /Deepfakes (1000)
            - /Face2Face (1000)
            - /FaceSwap (1000)
            - /NeuralTextures (1000)
        - /original_sequences
            - /actors (363)
                - /c23
                    - /videos
                        - .mp4 files
            - /youtube (1000)
            

## Folder Structure After ExtractFrame:
- /project
    - ExtractFrames.py here
    - /data
    - /Train
        - /Real
            - /Frames
                - /1
                    - .jpg files
                - /2
            - /Videos
                - 1.mp4
                - 2.mp4
        - /Fake
    - /Test
        - /Real
        - /Fake
