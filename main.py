from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from routers.ml import router as ml_router
from database import init_db
from routers import ml


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск
    print("🚀 Инициализация БД...")
    await init_db()
    print("✅ БД инициализирована")

    yield

    # Завершение
    print("👋 Завершение работы...")

app = FastAPI(
    title="ML Service",
    description="",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(ml.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to FastAPI ML Service",
        "documentation": "/docs",
        "version": "1.0.0"
    }


@app.get(
    "/health",
    summary="Health check / Проверка здоровья",
    description="Check if API is running / Проверить работу API"
)
async def health():
    """Health check endpoint / Эндпоинт проверки здоровья"""
    return {
        "status": "healthy / здоров",
        "database": "connected / подключена",
        "architecture": "DDD with async SQLAlchemy 2.0"
    }

if __name__ == "__main__":
    """
    Run application / Запустить приложение

    Usage / Использование:
        python main.py

    Then visit / Затем перейти на:
        http://localhost:8000/docs - Swagger UI
        http://localhost:8000/redoc - ReDoc
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        DDD User Management API with FastAPI                  ║
    ║        API управления пользователями с DDD                   ║
    ║                                                              ║
    ║  Architecture / Архитектура:                                 ║
    ║  ├─ API Layer (Routers) / Слой API                          ║
    ║  ├─ Service Layer / Слой сервисов                           ║
    ║  ├─ Unit of Work / Unit of Work                             ║
    ║  ├─ Repository Layer / Слой репозиториев                    ║
    ║  ├─ Domain Layer / Доменный слой                            ║
    ║  └─ Database / База данных                                  ║
    ║                                                              ║
    ║  Docs: http://localhost:8000/docs                            ║
    ║  ReDoc: http://localhost:8000/redoc                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
