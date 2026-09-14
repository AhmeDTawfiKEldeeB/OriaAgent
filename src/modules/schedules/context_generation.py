from datetime import datetime, time
from typing import Optional

from src.core.schedules import (
    FRIDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
)

SCHEDULES_BY_WEEKDAY = {
    0: MONDAY_SCHEDULE,
    1: TUESDAY_SCHEDULE,
    2: WEDNESDAY_SCHEDULE,
    3: THURSDAY_SCHEDULE,
    4: FRIDAY_SCHEDULE,
    5: SATURDAY_SCHEDULE,
    6: SUNDAY_SCHEDULE,
}


class ScheduleContextGenerator:
    """Generates the current activity context based on Oria's daily schedules."""

    @classmethod
    def get_current_activity(cls, current_time: Optional[datetime] = None) -> str:
        """Return Oria's current scheduled activity based on the current day and time."""
        now = current_time or datetime.now()
        day_schedule = SCHEDULES_BY_WEEKDAY.get(now.weekday(), {})
        current_time_val = now.time()

        for time_range, activity in day_schedule.items():
            start_str, end_str = time_range.split("-")
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
            start_time = time(start_h, start_m)
            end_time = time(end_h, end_m)

            # Normal interval within the same day
            if start_time <= end_time:
                if start_time <= current_time_val < end_time:
                    return activity
            else:
                # Interval spans across midnight (e.g. 23:00-06:00)
                if current_time_val >= start_time or current_time_val < end_time:
                    return activity

        return "Oria is taking a quiet moment to relax."
