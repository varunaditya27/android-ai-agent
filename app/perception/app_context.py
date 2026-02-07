"""
App Context Detector
====================

Detects app-specific states (e.g. YouTube video playing, ad showing)
and provides contextual hints to the agent so it can handle tricky
app interfaces without getting stuck.

Supported apps:
- YouTube: ad detection, video playback detection, search guidance
"""

from dataclasses import dataclass, field
from typing import Optional

from app.perception.ui_parser import UIElement
from app.utils.logger import get_logger

logger = get_logger(__name__)


# YouTube package names
YOUTUBE_PACKAGES = {
    "com.google.android.youtube",
    "com.google.android.youtube.tv",
    "com.google.android.apps.youtube.music",
}

# Media/video player packages (broader set)
MEDIA_PLAYER_PACKAGES = YOUTUBE_PACKAGES | {
    "com.google.android.apps.youtube.music",
    "com.spotify.music",
    "com.amazon.mp3",
    "org.videolan.vlc",
    "com.mxtech.videoplayer.ad",
    "com.mxtech.videoplayer.pro",
}


@dataclass
class AppContext:
    """Context about the current app state.

    Attributes:
        app_package: Current app package name.
        app_name: Human-readable app name.
        is_video_playing: Whether a video is currently playing.
        is_ad_playing: Whether an ad is currently playing.
        has_skip_button: Whether a "Skip Ad" button is visible.
        skip_button_element_id: Element ID of the skip button, if found.
        is_media_app: Whether the current app is a media player.
        is_search_field_focused: Whether a search/URL input field is focused.
        search_field_has_text: Whether the search field has text typed.
        context_hint: Guidance text for the agent.
    """

    app_package: str = ""
    app_name: str = ""
    is_video_playing: bool = False
    is_ad_playing: bool = False
    has_skip_button: bool = False
    skip_button_element_id: Optional[int] = None
    is_media_app: bool = False
    is_search_field_focused: bool = False
    search_field_has_text: bool = False
    context_hint: str = ""


class AppContextDetector:
    """Detects app-specific context to help the agent navigate tricky UIs."""

    # YouTube ad indicators (element text/content_desc patterns)
    AD_INDICATORS = [
        "skip ad", "skip ads", "skip in",
        "ad · ", "ad ·", "ad:", "advertisement",
        "visit advertiser", "learn more",
        "why this ad", "ad choices",
        "video will play after ad",
        "skip_ad_button",  # resource ID substring
        "ad_progress",
        "ad_text",
    ]

    # YouTube video playing indicators
    VIDEO_PLAYING_INDICATORS = [
        "player_view", "watch_player", "movie_player",
        "player_overlay", "player_control_play_pause",
        "time_bar", "player_control",
    ]

    # Video player controls (present when video is loaded/playing)
    PLAYER_CONTROL_INDICATORS = [
        "pause", "play", "fullscreen", "captions",
        "minimize player", "collapse",
        "player_collapse_button",
        "time_bar_current", "player_seekbar",
    ]

    # YouTube search-related
    YOUTUBE_SEARCH_INDICATORS = [
        "search_edit_text", "search_query",
        "search youtube", "search",
    ]

    # Google/Chrome search indicators
    GOOGLE_SEARCH_INDICATORS = [
        "search", "search_box", "query", "url_bar", "omnibox",
        "search or type url", "search or type web address",
        "google search", "search google",
    ]

    # Browser packages
    BROWSER_PACKAGES = [
        "com.android.chrome",
        "com.google.android.googlequicksearchbox",
        "org.mozilla.firefox",
        "com.microsoft.emmx",  # Edge
        "com.brave.browser",
    ]

    def detect(
        self,
        current_app: str,
        elements: list[UIElement],
        task: str = "",
    ) -> Optional[AppContext]:
        """Detect app-specific context.

        Args:
            current_app: Current app package name.
            elements: Current UI elements on screen.
            task: The user's task (for task-completion inference).

        Returns:
            AppContext if useful context detected, None otherwise.
        """
        if not current_app:
            return None

        # YouTube detection
        if any(pkg in current_app for pkg in YOUTUBE_PACKAGES):
            ctx = self._detect_youtube_context(current_app, elements, task)
            # Also check for search field in YouTube
            self._detect_search_field(ctx, current_app, elements)
            return ctx

        # Generic media app detection
        if any(pkg in current_app for pkg in MEDIA_PLAYER_PACKAGES):
            return self._detect_media_context(current_app, elements, task)

        # Browser/Google app detection (for search fields)
        if any(pkg in current_app for pkg in self.BROWSER_PACKAGES):
            ctx = AppContext(
                app_package=current_app,
                app_name=self._app_name_from_package(current_app),
            )
            self._detect_search_field(ctx, current_app, elements)
            if ctx.is_search_field_focused:
                return ctx

        return None

    def _detect_youtube_context(
        self,
        current_app: str,
        elements: list[UIElement],
        task: str,
    ) -> AppContext:
        """Detect YouTube-specific context."""
        ctx = AppContext(
            app_package=current_app,
            app_name="YouTube",
            is_media_app=True,
        )

        all_text = self._collect_element_text(elements)
        all_resource_ids = self._collect_resource_ids(elements)

        # --- Ad detection ---
        ad_detected = False
        skip_element_id = None

        for elem in elements:
            elem_text_lower = (
                (elem.text or "") + " " +
                (elem.content_desc or "") + " " +
                (elem.resource_id or "")
            ).lower()

            # Check for skip button
            if any(ind in elem_text_lower for ind in ["skip ad", "skip ads", "skip in", "skip_ad"]):
                ad_detected = True
                if elem.clickable:
                    skip_element_id = elem.index
                    logger.info(
                        "YouTube skip button detected",
                        element_id=elem.index,
                        text=elem.display_text,
                    )

            # Check for ad indicators
            if any(ind in elem_text_lower for ind in self.AD_INDICATORS):
                ad_detected = True

        ctx.is_ad_playing = ad_detected
        ctx.has_skip_button = skip_element_id is not None
        ctx.skip_button_element_id = skip_element_id

        # --- Video playing detection ---
        video_playing = self._is_video_playing(elements, all_resource_ids)
        ctx.is_video_playing = video_playing

        # --- Build context hint ---
        hints = []

        if ctx.is_ad_playing:
            if ctx.has_skip_button:
                hints.append(
                    f"📺 YOUTUBE AD PLAYING: A skippable ad is showing. "
                    f"Tap the 'Skip Ad' button (element [{ctx.skip_button_element_id}]) to skip it."
                )
            else:
                hints.append(
                    "📺 YOUTUBE AD PLAYING: An ad is showing but no skip button is available yet. "
                    "Use do(action=\"Wait\", seconds=5) to wait for the skip button to appear, "
                    "or wait for the ad to finish. Do NOT try to interact with ad elements."
                )
        elif ctx.is_video_playing:
            # Check if the task is a "play video" type task
            if self._is_play_task(task):
                hints.append(
                    "📺 VIDEO IS NOW PLAYING: The video/song is currently playing on YouTube. "
                    "Your task to play this video is COMPLETE. "
                    "Use finish(message=\"Video is now playing on YouTube\") to complete the task."
                )
            else:
                hints.append(
                    "📺 A video is currently playing on YouTube."
                )

        if hints:
            ctx.context_hint = "\n\n".join(hints)
            logger.info(
                "YouTube context detected",
                is_ad=ctx.is_ad_playing,
                has_skip=ctx.has_skip_button,
                is_playing=ctx.is_video_playing,
            )

        return ctx

    def _detect_media_context(
        self,
        current_app: str,
        elements: list[UIElement],
        task: str,
    ) -> Optional[AppContext]:
        """Detect generic media player context."""
        ctx = AppContext(
            app_package=current_app,
            app_name=self._app_name_from_package(current_app),
            is_media_app=True,
        )

        all_resource_ids = self._collect_resource_ids(elements)
        video_playing = self._is_video_playing(elements, all_resource_ids)
        ctx.is_video_playing = video_playing

        if video_playing and self._is_play_task(task):
            ctx.context_hint = (
                f"📺 MEDIA IS NOW PLAYING in {ctx.app_name}. "
                "Your task to play this media is COMPLETE. "
                f"Use finish(message=\"Media is now playing in {ctx.app_name}\") to complete the task."
            )
            return ctx

        return None

    def _is_video_playing(
        self,
        elements: list[UIElement],
        resource_ids: str,
    ) -> bool:
        """Check if a video/media is currently playing based on UI elements."""
        # Check resource IDs for player components
        player_resource_hits = sum(
            1 for ind in self.VIDEO_PLAYING_INDICATORS
            if ind in resource_ids
        )
        if player_resource_hits >= 2:
            return True

        # Check for player control elements
        control_hits = 0
        has_pause = False
        has_timebar = False

        for elem in elements:
            elem_text = (
                (elem.text or "") + " " +
                (elem.content_desc or "") + " " +
                (elem.resource_id or "")
            ).lower()

            if "pause" in elem_text:
                has_pause = True
                control_hits += 1
            if any(ind in elem_text for ind in ["time_bar", "seekbar", "progress"]):
                has_timebar = True
                control_hits += 1
            if any(ind in elem_text for ind in self.PLAYER_CONTROL_INDICATORS):
                control_hits += 1

        # Video is likely playing if we see a pause button + time bar or 3+ control elements
        return (has_pause and has_timebar) or control_hits >= 3

    def _is_play_task(self, task: str) -> bool:
        """Check if the user's task is about playing media."""
        task_lower = task.lower()
        play_keywords = [
            "play", "watch", "listen", "open video", "open song",
            "put on", "stream", "start video", "start song",
            "play video", "play song", "play music",
            "youtube video", "youtube song",
        ]
        return any(kw in task_lower for kw in play_keywords)

    def _collect_element_text(self, elements: list[UIElement]) -> str:
        """Collect all text from elements into a single lowercase string."""
        parts = []
        for elem in elements:
            if elem.text:
                parts.append(elem.text)
            if elem.content_desc:
                parts.append(elem.content_desc)
        return " ".join(parts).lower()

    def _collect_resource_ids(self, elements: list[UIElement]) -> str:
        """Collect all resource IDs into a single lowercase string."""
        parts = []
        for elem in elements:
            if elem.resource_id:
                parts.append(elem.resource_id)
        return " ".join(parts).lower()

    def _app_name_from_package(self, package: str) -> str:
        """Extract a human-readable app name from a package name."""
        name_map = {
            "com.google.android.youtube": "YouTube",
            "com.google.android.apps.youtube.music": "YouTube Music",
            "com.spotify.music": "Spotify",
            "com.amazon.mp3": "Amazon Music",
            "org.videolan.vlc": "VLC",
            "com.android.chrome": "Chrome",
            "com.google.android.googlequicksearchbox": "Google",
            "org.mozilla.firefox": "Firefox",
        }
        for pkg, name in name_map.items():
            if pkg in package:
                return name
        # Fallback: extract last part of package
        parts = package.split(".")
        return parts[-1].title() if parts else package

    def _detect_search_field(
        self,
        ctx: AppContext,
        current_app: str,
        elements: list[UIElement],
    ) -> None:
        """Detect if a search/URL field is focused with text typed.
        
        Updates the AppContext in-place with search field state and adds hints.
        """
        # Determine which search indicators to use
        is_youtube = any(pkg in current_app for pkg in YOUTUBE_PACKAGES)
        is_browser = any(pkg in current_app for pkg in self.BROWSER_PACKAGES)
        
        indicators = self.YOUTUBE_SEARCH_INDICATORS if is_youtube else self.GOOGLE_SEARCH_INDICATORS
        
        for elem in elements:
            elem_text = (
                (elem.text or "") + " " +
                (elem.content_desc or "") + " " +
                (elem.resource_id or "")
            ).lower()
            
            # Check if this is a search field
            is_search_field = any(ind in elem_text for ind in indicators)
            
            if is_search_field and elem.focused:
                ctx.is_search_field_focused = True
                # Check if there's text in the field
                if elem.text and len(elem.text.strip()) > 0:
                    ctx.search_field_has_text = True
                    
                    # Add hint to press Enter
                    enter_hint = (
                        f"🔍 SEARCH FIELD READY: You've typed '{elem.text}' in the search field. "
                        "To submit the search, use do(action=\"PressKey\", key=\"enter\"). "
                        "Do NOT wait for suggestions — press Enter immediately."
                    )
                    
                    # Append or set hint
                    if ctx.context_hint:
                        ctx.context_hint += "\n\n" + enter_hint
                    else:
                        ctx.context_hint = enter_hint
                    
                    logger.info(
                        "Search field with text detected",
                        app=ctx.app_name,
                        text=elem.text[:20],
                    )
                    return
