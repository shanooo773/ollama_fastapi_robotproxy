# AI-Powered Robotic Proxy for Remote Business Presence in Expos

This is a Final Year Design Project (FYDP) developed in the Department of Computer Engineering,
University of Engineering and Technology (UET) Taxila.

## What the Project Does

Business exhibitions and trade expos provide valuable opportunities for networking, product
demonstrations, and collaboration. However, geographical distance, travel costs, visa
restrictions, and scheduling conflicts often prevent individuals and organizations from
participating in these events.

This project addresses that problem by developing an AI-powered robotic telepresence system
that lets a remote user attend exhibitions through an internet-connected mobile robot. Beyond
conventional telepresence (video/audio), the system includes an on-device AI assistant capable
of presenting company information, answering frequently asked questions using a
Retrieval-Augmented Generation (RAG) pipeline, and automatically generating structured summaries
of business interactions.

The system combines a mobile robotic platform, onboard/edge computing (NVIDIA Jetson), ROS2-based
robot control, camera and audio interfaces, an LLM/RAG-based AI assistant, speech interaction,
backend APIs, and a reporting dashboard.

## Project Objectives

- Design a remotely operable mobile telepresence robot.
- Develop a low-latency audio and video communication system.
- Integrate a local Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) for
  intelligent business assistance.
- Implement Speech-to-Text (STT) and Text-to-Speech (TTS) modules for natural communication.
- Record and summarize conversations for post-event analysis.
- Develop a dashboard for interaction reports and potential lead generation.
- Ensure reliable and scalable system performance suitable for real-world deployment.

## Academic Information

- **Department**: Computer Engineering, University of Engineering and Technology (UET) Taxila
- **Supervisor**: Dr. Muhammad Haroon Yousaf
- **Industry Advisor**: Engr. Saran Khaliq, BrainSwarm Robotics Pvt. Ltd.

## Team

| Team | Member | Registration No. | Focus |
|---|---|---|---|
| Team Alpha | M Farhat Mehdi | 23-CP-43 | Robot & Telepresence (ROS2, hardware, teleoperation) |
| Team Alpha | Umama Hanif | 23-CP-19 | Robot & Telepresence platform support |
| Team Beta | Shayan Humayun | 23-CP-69 | AI Stack & Backend (FastAPI, Ollama, RAG) |
| Team Beta | Aqib Ali | 23-CP-47 | AI Stack support, model routing |

## Repositories

- Documentation repository: https://github.com/shanooo773/Robot_proxy
- AI backend repository: https://github.com/shanooo773/ollama_fastapi_robotproxy
