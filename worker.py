#!/usr/bin/env python3
"""
RunPod serverless worker for Coqui TTS (XTTS)
Starts the RunPod serverless handler with concurrency control
"""

import runpod
import concurrency
import handler

# Start RunPod serverless
runpod.serverless.start({
    "handler": handler.handler,
    "concurrency_modifier": concurrency.adjust_concurrency,
})
