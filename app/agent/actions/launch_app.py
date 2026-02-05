"""
Launch App Action
=================

App launching with package name resolution and fuzzy matching.
"""

from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Common app name to package name mappings
APP_PACKAGES = {
    # Google Apps
    "youtube": "com.google.android.youtube",
    "chrome": "com.android.chrome",
    "gmail": "com.google.android.gm",
    "google maps": "com.google.android.apps.maps",
    "maps": "com.google.android.apps.maps",
    "google photos": "com.google.android.apps.photos",
    "photos": "com.google.android.apps.photos",
    "play store": "com.android.vending",
    "google play": "com.android.vending",
    "google": "com.google.android.googlequicksearchbox",
    "google assistant": "com.google.android.apps.googleassistant",
    "google drive": "com.google.android.apps.docs",
    "drive": "com.google.android.apps.docs",
    "google calendar": "com.google.android.calendar",
    "calendar": "com.google.android.calendar",
    "google keep": "com.google.android.keep",
    "keep": "com.google.android.keep",
    "google meet": "com.google.android.apps.meetings",
    "meet": "com.google.android.apps.meetings",
    "google duo": "com.google.android.apps.tachyon",
    "duo": "com.google.android.apps.tachyon",

    # System Apps
    "settings": "com.android.settings",
    "camera": "com.android.camera2",
    "phone": "com.android.dialer",
    "dialer": "com.android.dialer",
    "contacts": "com.android.contacts",
    "messages": "com.google.android.apps.messaging",
    "sms": "com.google.android.apps.messaging",
    "clock": "com.google.android.deskclock",
    "alarm": "com.google.android.deskclock",
    "calculator": "com.google.android.calculator",
    "files": "com.google.android.apps.nbu.files",
    "file manager": "com.google.android.apps.nbu.files",

    # Social Media
    "whatsapp": "com.whatsapp",
    "instagram": "com.instagram.android",
    "facebook": "com.facebook.katana",
    "messenger": "com.facebook.orca",
    "twitter": "com.twitter.android",
    "x": "com.twitter.android",
    "tiktok": "com.zhiliaoapp.musically",
    "snapchat": "com.snapchat.android",
    "linkedin": "com.linkedin.android",
    "reddit": "com.reddit.frontpage",
    "discord": "com.discord",
    "telegram": "org.telegram.messenger",

    # Entertainment
    "spotify": "com.spotify.music",
    "netflix": "com.netflix.mediaclient",
    "amazon prime": "com.amazon.avod.thirdpartyclient",
    "prime video": "com.amazon.avod.thirdpartyclient",
    "disney+": "com.disney.disneyplus",
    "disney plus": "com.disney.disneyplus",
    "hulu": "com.hulu.plus",
    "hbo max": "com.wbd.stream",
    "twitch": "tv.twitch.android.app",

    # Productivity
    "chatgpt": "com.openai.chatgpt",
    "slack": "com.Slack",
    "zoom": "us.zoom.videomeetings",
    "microsoft teams": "com.microsoft.teams",
    "teams": "com.microsoft.teams",
    "notion": "notion.id",
    "evernote": "com.evernote",
    "trello": "com.trello",

    # Shopping
    "amazon": "com.amazon.mShop.android.shopping",
    "amazon shopping": "com.amazon.mShop.android.shopping",
    "ebay": "com.ebay.mobile",
    "walmart": "com.walmart.android",
    "target": "com.target.ui",

    # Finance
    "paypal": "com.paypal.android.p2pmobile",
    "venmo": "com.venmo",
    "cash app": "com.squareup.cash",

    # Travel
    "uber": "com.ubercab",
    "lyft": "me.lyft.android",
    "airbnb": "com.airbnb.android",
    "booking": "com.booking",

    # News
    "news": "com.google.android.apps.magazines",
    "google news": "com.google.android.apps.magazines",
    "cnn": "com.cnn.mobile.android.phone",
    "bbc news": "bbc.mobile.news.ww",
}


def resolve_package_name(app_name: str) -> str:
    """
    Resolve an app name to its package name.

    Args:
        app_name: Common app name or package name.

    Returns:
        Package name (original if not found in mapping).
    """
    # Check if it's already a package name
    if "." in app_name and not " " in app_name:
        return app_name

    # Normalize and lookup
    normalized = app_name.lower().strip()

    # Direct match
    if normalized in APP_PACKAGES:
        return APP_PACKAGES[normalized]

    # Fuzzy match - find best match
    best_match = None
    best_score = 0

    for name, package in APP_PACKAGES.items():
        # Check if app_name is contained in the key
        if normalized in name:
            score = len(normalized) / len(name)
            if score > best_score:
                best_score = score
                best_match = package

        # Check if key is contained in app_name
        if name in normalized:
            score = len(name) / len(normalized)
            if score > best_score:
                best_score = score
                best_match = package

    if best_match and best_score > 0.5:
        logger.debug(
            "Fuzzy matched app name",
            input=app_name,
            package=best_match,
            score=best_score,
        )
        return best_match

    # Return original (might be a package name already)
    logger.warning(
        "Could not resolve app name, using as-is",
        app_name=app_name,
    )
    return app_name


async def launch_app(
    device: CloudDevice,
    app_name: str,
) -> bool:
    """
    Launch an app by name.

    Args:
        device: Cloud device.
        app_name: App name or package name.

    Returns:
        True if launch succeeded.
    """
    package_name = resolve_package_name(app_name)

    logger.info(
        "Launching app",
        app_name=app_name,
        package=package_name,
    )

    result = await device.launch_app(package_name)
    return result.success


async def get_current_app(device: CloudDevice) -> str:
    """
    Get the currently active app.

    Args:
        device: Cloud device.

    Returns:
        Package name of current app.
    """
    return await device.get_current_app()


def get_app_name_from_package(package_name: str) -> Optional[str]:
    """
    Get a friendly app name from package name.

    Args:
        package_name: Android package name.

    Returns:
        Friendly name if found.
    """
    # Reverse lookup
    for name, package in APP_PACKAGES.items():
        if package == package_name:
            return name.title()

    # Extract from package name
    parts = package_name.split(".")
    if len(parts) > 1:
        return parts[-1].replace("_", " ").title()

    return None
