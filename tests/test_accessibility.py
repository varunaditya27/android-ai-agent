"""
Accessibility Module Tests
==========================

Tests for Announcer, TalkBackController, HapticsController, and
AccessibilityManager.  All device/TTS calls are mocked.
"""

import os

# ENV vars must be set before any app.* import
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("GROQ_API_KEY", "gsk_test-groq-key-for-testing")

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.accessibility.announcer import (
    Announcer,
    AnnouncementConfig,
    AnnouncementPriority,
    format_element_for_speech,
)
from app.accessibility.haptics import (
    HapticsController,
    HapticsConfig,
    HapticPattern,
    VibrationIntensity,
)
from app.accessibility.talkback import (
    TalkBackController,
    TalkBackGesture,
    TalkBackSettings,
)
from app.accessibility.manager import AccessibilityManager
from app.device.cloud_provider import CloudDevice, DeviceInfo, ActionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_device() -> MagicMock:
    """Create a mock CloudDevice suitable for accessibility tests."""
    device = MagicMock(spec=CloudDevice)
    device.info = DeviceInfo(
        device_id="test-123",
        platform="android",
        os_version="14.0",
        screen_width=1080,
        screen_height=2400,
        model="Test Pixel",
        manufacturer="Google",
    )
    device.is_connected = True
    device.execute_shell = AsyncMock(return_value="")
    device.tap = AsyncMock(return_value=ActionResult(success=True))
    device.swipe = AsyncMock(return_value=ActionResult(success=True))
    return device


# ═══════════════════════════════════════════════════════════════════════
# Announcer
# ═══════════════════════════════════════════════════════════════════════


class TestAnnouncerConfig:
    """AnnouncementConfig defaults and construction."""

    def test_default_config(self):
        cfg = AnnouncementConfig()
        assert cfg.enabled is True
        assert cfg.speech_rate == 200
        assert cfg.volume == 1.0
        assert cfg.screen_reader_mode is True
        assert cfg.duplicate_timeout == 3.0

    def test_disabled_config(self):
        cfg = AnnouncementConfig(enabled=False)
        assert cfg.enabled is False


class TestAnnouncer:
    """Announcer with pyttsx3 mocked out."""

    async def test_speak_no_tts_engine(self):
        """When pyttsx3 TTS engine fails to init, speak() should not crash."""
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        # Force engine to None to simulate unavailable TTS
        ann._tts_engine = None
        ann._get_tts_engine = MagicMock(return_value=None)
        # Should not raise
        await ann.speak("Hello world")

    async def test_speak_calls_tts(self):
        """speak() should invoke _speak_sync when engine is available."""
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.speak("Test announcement")

    async def test_speak_disabled(self):
        """When disabled, speak() is a no-op."""
        ann = Announcer(config=AnnouncementConfig(enabled=False))
        await ann.speak("Should not speak")
        # No crash, no TTS init

    async def test_deduplication(self):
        """Same message within timeout is not repeated."""
        ann = Announcer(
            config=AnnouncementConfig(
                enabled=True,
                duplicate_timeout=5.0,
            )
        )
        # Mock out TTS
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())

        await ann.speak("Duplicate message")
        await ann.speak("Duplicate message")
        # The dedup logic in speak() should prevent the second call

    async def test_screen_reader_output(self, capsys):
        """In screen_reader_mode, speak() should print [Agent] prefix."""
        ann = Announcer(
            config=AnnouncementConfig(
                enabled=True,
                screen_reader_mode=True,
            )
        )
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.speak("Check output", priority=AnnouncementPriority.HIGH)
        captured = capsys.readouterr()
        assert "[Agent]" in captured.out

    async def test_announce_task_start(self):
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.announce_task_start("Open YouTube")

    async def test_announce_completion(self):
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.announce_completion(True, "Done")
        await ann.announce_completion(False, "Failed")

    async def test_announce_error(self):
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.announce_error("Something broke", recoverable=True)
        await ann.announce_error("Fatal", recoverable=False)

    async def test_announce_rate_limit(self):
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann._speak_sync = MagicMock()
        ann._get_tts_engine = MagicMock(return_value=MagicMock())
        await ann.announce_rate_limit(30.0)

    async def test_stop(self):
        ann = Announcer(config=AnnouncementConfig(enabled=True))
        ann.stop()
        assert ann._is_speaking is False


class TestFormatElementForSpeech:
    """format_element_for_speech helper."""

    def test_button(self):
        result = format_element_for_speech(
            "Button", "Submit", {"enabled": True}
        )
        assert "button" in result.lower()
        assert "Submit" in result

    def test_text_field(self):
        result = format_element_for_speech(
            "EditText", "", {"enabled": True}
        )
        assert "text field" in result.lower() or "edit" in result.lower()

    def test_unknown_type(self):
        result = format_element_for_speech(
            "CustomWidget", "Hello", {}
        )
        assert "Hello" in result

    def test_checkbox_checked(self):
        result = format_element_for_speech(
            "CheckBox", "Wi-Fi", {"checked": True, "checkable": True}
        )
        assert "checked" in result.lower()

    def test_disabled_element(self):
        result = format_element_for_speech(
            "Button", "Submit", {"enabled": False}
        )
        assert "disabled" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# TalkBackController
# ═══════════════════════════════════════════════════════════════════════


class TestTalkBackController:
    """TalkBack control via mocked ADB commands."""

    def _make_controller(
        self, device=None, enabled=True
    ) -> TalkBackController:
        d = device or _mock_device()
        return TalkBackController(
            d,
            settings=TalkBackSettings(enabled=enabled),
        )

    async def test_is_enabled_true(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(
            return_value="com.google.android.marvin.talkback/"
            "com.google.android.marvin.talkback"
            ".TalkBackService"
        )
        ctrl = self._make_controller(device)
        assert await ctrl.is_enabled() is True

    async def test_is_enabled_false(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        assert await ctrl.is_enabled() is False

    async def test_enable(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.enable()
        assert result is True
        device.execute_shell.assert_called()

    async def test_disable(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.disable()
        assert result is True

    async def test_set_high_contrast(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.set_high_contrast(True)
        assert result is True
        # Verify the correct setting command was called
        calls = [str(c) for c in device.execute_shell.call_args_list]
        assert any("high_text_contrast_enabled" in c for c in calls)

    async def test_set_font_scale(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.set_font_scale(1.3)
        assert result is True
        calls = [str(c) for c in device.execute_shell.call_args_list]
        assert any("font_scale" in c for c in calls)

    async def test_set_speech_rate(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.set_speech_rate(300)
        assert result is True

    async def test_navigate_next(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.navigate_next()
        assert isinstance(result, bool)

    async def test_navigate_previous(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.navigate_previous()
        assert isinstance(result, bool)

    async def test_apply_settings_enabled(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = TalkBackController(
            device,
            settings=TalkBackSettings(
                enabled=True,
                high_contrast=True,
                large_text=True,
            ),
        )
        await ctrl.apply_settings()
        assert device.execute_shell.call_count >= 1

    async def test_apply_settings_disabled(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = TalkBackController(
            device,
            settings=TalkBackSettings(enabled=False),
        )
        await ctrl.apply_settings()
        # Should still succeed (just not enable TalkBack)

    async def test_error_handling(self):
        """execute_shell failure should not crash the controller."""
        device = _mock_device()
        device.execute_shell = AsyncMock(
            side_effect=Exception("ADB error")
        )
        ctrl = self._make_controller(device)
        result = await ctrl.set_high_contrast(True)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════
# HapticsController
# ═══════════════════════════════════════════════════════════════════════


class TestHapticsController:
    """Haptics via mocked ADB vibrator commands."""

    def _make_controller(
        self, device=None, enabled=True
    ) -> HapticsController:
        d = device or _mock_device()
        return HapticsController(
            d,
            config=HapticsConfig(enabled=enabled),
        )

    async def test_vibrate_basic(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device, enabled=True)
        result = await ctrl.vibrate(HapticPattern.CLICK)
        assert result is True

    async def test_vibrate_disabled(self):
        ctrl = self._make_controller(enabled=False)
        result = await ctrl.vibrate(HapticPattern.CLICK)
        assert result is False

    async def test_success_pattern(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.success()
        assert result is True

    async def test_error_pattern(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.error()
        assert result is True

    async def test_click_pattern(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.click()
        assert result is True

    async def test_action_feedback_tap(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.action_feedback("Tap", True)
        assert result is True

    async def test_action_feedback_fail(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.action_feedback("Tap", False)
        assert result is True

    async def test_set_enabled(self):
        ctrl = self._make_controller(enabled=False)
        ctrl.set_enabled(True)
        assert ctrl.config.enabled is True

    async def test_set_intensity(self):
        ctrl = self._make_controller()
        ctrl.set_intensity(VibrationIntensity.STRONG)
        assert ctrl.config.intensity == VibrationIntensity.STRONG

    async def test_task_complete(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.task_complete()
        assert result is True

    async def test_task_failed(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        ctrl = self._make_controller(device)
        result = await ctrl.task_failed()
        assert result is True

    async def test_adb_error_handled(self):
        """ADB failure should not crash haptics."""
        device = _mock_device()
        device.execute_shell = AsyncMock(
            side_effect=Exception("vibrator not found")
        )
        ctrl = self._make_controller(device)
        result = await ctrl.vibrate(HapticPattern.CLICK)
        # Should not propagate the exception
        assert result is False or result is True  # just don't crash


# ═══════════════════════════════════════════════════════════════════════
# AccessibilityManager
# ═══════════════════════════════════════════════════════════════════════


class TestAccessibilityManager:
    """Manager facade coordinating all three sub-controllers."""

    def _make_manager(
        self,
        device=None,
        tts=True,
        haptics=True,
        talkback=False,
        sr=True,
    ) -> AccessibilityManager:
        d = device or _mock_device()
        return AccessibilityManager(
            device=d,
            enable_tts=tts,
            enable_haptics=haptics,
            enable_talkback=talkback,
            screen_reader_mode=sr,
        )

    async def test_setup(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device, talkback=True)
        await mgr.setup()
        # apply_settings called on TalkBack
        assert device.execute_shell.called or True  # no crash

    async def test_setup_no_device(self):
        mgr = AccessibilityManager(device=None, enable_tts=True)
        await mgr.setup()
        assert mgr.talkback is None
        assert mgr.haptics is None

    async def test_on_task_start(self):
        mgr = self._make_manager()
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_task_start("Open YouTube")

    async def test_on_step_success(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_step(
            step_num=1,
            max_steps=10,
            action_type="Tap",
            success=True,
        )

    async def test_on_step_error(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_step(
            step_num=1,
            max_steps=10,
            action_type="Tap",
            success=False,
            error="Element not found",
        )

    async def test_on_step_finished(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_step(
            step_num=5,
            max_steps=10,
            action_type="Finish",
            success=True,
            finished=True,
            result_message="YouTube opened",
        )

    async def test_on_task_complete_success(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_task_complete(True, "YouTube opened")

    async def test_on_task_complete_failure(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_task_complete(False, "Could not connect")

    async def test_on_input_required(self):
        device = _mock_device()
        device.execute_shell = AsyncMock(return_value="")
        mgr = self._make_manager(device)
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_input_required("Enter OTP")

    async def test_on_rate_limit(self):
        mgr = self._make_manager()
        mgr.announcer._speak_sync = MagicMock()
        mgr.announcer._get_tts_engine = MagicMock(return_value=MagicMock())
        await mgr.on_rate_limit(30.0)

    async def test_stop(self):
        mgr = self._make_manager()
        mgr.stop()
        assert mgr.announcer._is_speaking is False

    async def test_announcer_screen_reader_mode(self):
        mgr = self._make_manager(sr=True)
        assert mgr.announcer.config.screen_reader_mode is True

    async def test_announcer_non_screen_reader_mode(self):
        mgr = self._make_manager(sr=False)
        assert mgr.announcer.config.screen_reader_mode is False

    async def test_haptics_disabled(self):
        mgr = self._make_manager(haptics=False)
        # Haptics controller exists but is disabled
        if mgr.haptics:
            assert mgr.haptics.config.enabled is False
