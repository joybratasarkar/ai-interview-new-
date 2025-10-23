# apps/ai_interview/utils/task_poller.py

import asyncio
import json
import logging
from celery.result import AsyncResult
from apps.ai_interview.celery_app import celery_app
from apps.ai_interview.services.shared import manager

logger = logging.getLogger(__name__)

async def poll_task_result(task_id: str, client_id: str, task_type: str, timeout: int = 10):
    """
    Poll for Celery task result and send directly to WebSocket when ready
    
    Args:
        task_id: Celery task ID
        client_id: WebSocket client/room ID  
        task_type: Type of task (audio, video, answer)
        timeout: Maximum time to wait in seconds
    """
    logger.info(f"🔄 POLLER STARTED - Task: {task_id}, Client: {client_id}, Type: {task_type}")
    result = AsyncResult(task_id, app=celery_app)
    
    try:
        # Poll for result with timeout
        for i in range(timeout * 10):  # Check every 0.1 seconds
            logger.debug(f"🔍 Polling attempt {i+1} for task {task_id}")
            
            if result.ready():
                logger.info(f"✅ TASK READY - {task_id}")
                
                if result.successful():
                    task_result = result.result
                    logger.info(f"🎯 TASK SUCCESSFUL - {task_id}, Result: {task_result}")
                    
                    if task_result:  # Task returned a result
                        logger.info(f"📤 SENDING TO WEBSOCKET - Client: {client_id}, Result: {task_result}")
                        
                        # Send result directly to WebSocket
                        await manager.send_to_room(
                            client_id, 
                            json.dumps(task_result)
                        )
                        logger.info(f"✅ SENT TO WEBSOCKET - Task {task_id} result delivered to {client_id}")
                        return task_result
                    else:
                        logger.warning(f"❌ TASK COMPLETED BUT NO RESULT - {task_id} for {client_id}")
                        return None
                else:
                    # Task failed
                    logger.error(f"💥 TASK FAILED - {task_id}")
                    error_msg = {
                        "room_id": client_id,
                        "task_type": task_type,
                        "error": str(result.result) if result.result else "Task failed",
                        "task_id": task_id
                    }
                    logger.error(f"💥 TASK ERROR - {task_id} failed for {client_id}: {error_msg}")
                    
                    await manager.send_to_room(
                        client_id,
                        json.dumps(error_msg)
                    )
                    logger.error(f"💥 ERROR SENT TO WEBSOCKET - {task_id}")
                    return None
                    
            await asyncio.sleep(0.1)
        
        # Timeout reached
        timeout_msg = {
            "room_id": client_id,
            "task_type": task_type,
            "error": f"Task {task_id} timed out after {timeout}s",
            "task_id": task_id
        }
        logger.warning(f"Task {task_id} timed out for {client_id}")
        
        await manager.send_to_room(
            client_id,
            json.dumps(timeout_msg)
        )
        
    except Exception as e:
        error_msg = {
            "room_id": client_id,
            "task_type": task_type,
            "error": f"Error polling task {task_id}: {str(e)}",
            "task_id": task_id
        }
        logger.error(f"Error polling task {task_id} for {client_id}: {e}")
        
        await manager.send_to_room(
            client_id,
            json.dumps(error_msg)
        )
    
    return None


async def send_task_immediate(task_id: str, client_id: str, task_type: str):
    """
    Send immediate acknowledgment that task was queued
    """
    ack_msg = {
        "room_id": client_id,
        "task_type": task_type,
        "task_id": task_id,
        "status": "queued"
    }
    
    await manager.send_to_room(
        client_id,
        json.dumps(ack_msg)
    )