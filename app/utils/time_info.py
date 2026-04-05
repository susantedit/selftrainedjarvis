from datetime import datetime


def get_time_information() -> str:
    now = datetime.now()
    day = now.strftime("%A")
    date = now.strftime("%B %d, %Y")
    time = now.strftime("%I:%M %p").lstrip("0")

    # ordinal suffix
    d = now.day
    suffix = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    date_ord = now.strftime(f"%B {d}{suffix}, %Y")

    return f"{day}, {date_ord} {time}"
