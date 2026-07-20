# Week 1 Progress: Platform Bring-up and AI Environment Preparation

## Team Alpha (Robot & Telepresence)

- Prepared an Ubuntu development environment for ROS2 practice, since the Jetson Xavier NX
  hardware was not yet available.
- Practiced ROS2 fundamentals (nodes, topics, publishers/subscribers, services, actions,
  parameters, launch files) using the ROS2 turtlesim simulator.
- Verified keyboard teleoperation at the simulation level and inspected ROS2 topics
  (`ros2 topic list`, `ros2 topic echo`).
- Built a basic Python obstacle-avoidance simulation as a learning exercise.
- Documented (but could not yet execute) the Xavier NX setup procedure, NVIDIA SDK Manager
  workflow, remote SSH access workflow, and LIDAR verification workflow, since the physical
  hardware had not been delivered.
- Built digital twin simulation prototypes (Python + Pygame) of an expo robot and a hospital
  delivery robot, demonstrating movement, obstacle handling, mission selection, emergency stop,
  and dashboard-style monitoring concepts.
- Created and organized the shared GitHub repository structure.

## Team Beta (AI Stack & Backend)

- Designed the AI backend architecture: FastAPI + Ollama, with a remote-first model routing
  strategy and local fallback for reliability.
- Planned the REST API surface: `/health` (service/model/CUDA status), `/chat` (send a message,
  get a model response), and `/agent` (planned for future tool-use/task execution).
- Designed the model routing policy: try the primary remote model, fall back to a secondary
  remote model, then fall back to a small local model if all remote options fail.
- Planned CUDA/GPU-aware health checks so the service reports GPU availability without failing
  when no GPU is present.
- Identified the required Python dependencies (FastAPI, Uvicorn, httpx, Pydantic,
  pydantic-settings, python-dotenv) and defined an environment-variable-based configuration
  scheme (`.env`) for model names, endpoints, and timeouts.

## Problems Faced in Week 1

- Xavier NX and LIDAR hardware were not available, so several hardware tasks were documented
  as planned procedures rather than completed work.
- Remote SSH testing could not be completed without the target Jetson device.
- ROS2 distribution choice depends on the eventual Jetson's Ubuntu/JetPack version, which was
  not yet known.
