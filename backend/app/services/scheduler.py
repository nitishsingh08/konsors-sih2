import logging
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text
from app.core.config import settings
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

def poll_firms_nrt_feed():
    """
    Scheduled job: Polls NASA FIRMS active thermal detections every 15 minutes.
    Guarded by PostgreSQL session advisory lock to prevent double-polling across workers.
    """
    db = SessionLocal()
    lock_acquired = False
    try:
        # 1. Concurrency guard using Postgres advisory lock
        if not settings.DATABASE_URL.startswith("sqlite"):
            result = db.execute(text(f"SELECT pg_try_advisory_lock({settings.ADVISORY_LOCK_ID})")).scalar()
            if not result:
                logger.info("Advisory lock held by another worker. Skipping FIRMS polling in this process.")
                return
            lock_acquired = True

        logger.info("Executing scheduled NASA FIRMS NRT feed synchronization...")
        # Polling logic: fetch FIRMS CSV/JSON -> DBSCAN clustering -> heuristic classification -> DB commit
        # (Mock placeholder logic for demo resilience)

    except Exception as e:
        logger.error(f"Error during scheduled FIRMS poll: {e}")
    finally:
        if lock_acquired:
            try:
                db.execute(text(f"SELECT pg_advisory_unlock({settings.ADVISORY_LOCK_ID})"))
            except Exception as e:
                logger.warning(f"Error unlocking advisory lock: {e}")
        db.close()

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(
            poll_firms_nrt_feed,
            'interval',
            minutes=settings.POLL_INTERVAL_MINUTES,
            id='firms_nrt_poller',
            replace_existing=True
        )
        scheduler.start()
        logger.info(f"APScheduler started: polling FIRMS every {settings.POLL_INTERVAL_MINUTES} minutes.")

def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")
