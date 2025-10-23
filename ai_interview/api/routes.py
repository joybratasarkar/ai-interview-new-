from fastapi import APIRouter
from apps.ai_interview.tasks.example import add
from apps.ai_interview.celery_app import celery_app
from celery.result import AsyncResult
from fastapi.responses import JSONResponse
router = APIRouter()

@router.post("/add")
async def add_numbers(x: int, y: int):
    task = add.delay(x, y)
    return {"task_id": task.id}



router = APIRouter()

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if not res:
        
        return  JSONResponse(
    status_code=404,
    content={"status_code": 404, "detail": "Task not found"}
)
    return {
        "id":     task_id,
        "status": res.status,
        "result": res.result if res.ready() else None
    }
