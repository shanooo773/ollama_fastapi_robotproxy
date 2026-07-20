# Technology Stack

## Robotics & Hardware

- NVIDIA Jetson (Nano / Xavier NX class edge compute)
- ROS 2 (robot control, sensor communication)
- Wheeled mobile robot platform, built on an AION Robotics UGV chassis with a Cube/Pixhawk
  autopilot running ArduRover, plus a HERE3 GPS/Compass module
- RPLIDAR S2 (LIDAR sensor) for obstacle detection and mapping
- RGB camera for the telepresence video feed
- Microphone array and speaker system for voice interaction

## Artificial Intelligence

- Ollama, for running local Large Language Models (LLMs)
- Candidate LLMs benchmarked: Llama 3.1 (8B), Mistral (7B), Phi-3 Mini (3.8B), Qwen2 (7B)
- Retrieval-Augmented Generation (RAG) pipeline for grounded, factual answers about the project
  and company being represented
- ChromaDB as the vector database for retrieval
- nomic-embed-text as the embedding model
- Speech-to-Text (STT) and Text-to-Speech (TTS) for the Jetson-side voice interface

## Backend & APIs

- Python
- FastAPI, exposing REST endpoints (`/health`, `/chat`, `/agent`, `/rag/ask`) and a streaming
  WebSocket endpoint for real-time conversation
- httpx for async HTTP communication with Ollama

## Communication & Networking

- WebRTC / real-time audio-video streaming for the telepresence feed
- Tailscale, used to connect the robot (Jetson) and the GPU-hosted AI backend securely without
  port forwarding, regardless of physical location

## Development Environment

- CUDA Toolkit (for GPU-accelerated inference on the workstation hosting Ollama)
- Python virtual environments
- Git & GitHub for version control
