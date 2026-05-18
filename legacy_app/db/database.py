from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from legacy_app.db.models import Base


# SQLAlchemy 2.0: Используем async engine для асинхронных операций
DATABASE_URL = "sqlite+aiosqlite:///legacy_app/db/ml.db"

# Создаем async engine с echo для отладки SQL
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Печатать SQL запросы
    future=True  # Использовать API SQLAlchemy 2.0
)

# Фабрика сессий для создания сессий базы данных
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Сохранять объекты после коммита
    autoflush=False,  # Ручной контроль над flush
    autocommit=False  # Явный контроль транзакций
)


async def get_session() -> AsyncSession:
    """
    Зависимость для получения сессии БД

    Использование в FastAPI:
        @app.get("/users")
        async def get_users(session: AsyncSession = Depends(get_session)):
            ...

    Yields:
        AsyncSession: Сессия базы данных
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """
    Инициализация таблиц базы данных

    Создает все таблицы, определенные в Base.metadata
    """
    async with engine.begin() as conn:
        # Удалить все таблицы (только для разработки!)
        await conn.run_sync(Base.metadata.drop_all)

        # Create all tables / Создать все таблицы
        await conn.run_sync(Base.metadata.create_all)
