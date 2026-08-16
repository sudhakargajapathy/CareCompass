"""Unit tests for the DEMO_VIDEO_URL Loom embed (helper, config knob, wiring).

The contract under test: when the env knob holds a Loom link, the app renders
a dedicated "Watch the demo" expander ABOVE "How it works"; when the knob is
unset or holds anything that is not a Loom video link, that expander does not
exist at all — no placeholder, no empty player — and "How it works" renders
byte-identically either way.
"""

import inspect
from unittest.mock import MagicMock

import pytest

import app as app_module
from app import _demo_video_embed_url, _render_demo_video, render_header
from utils.config import Config


class TestEmbedUrlNormalization:
    def test_share_link_normalizes_to_embed(self):
        # The Share button copies /share/<id>; the player iframe loads
        # /embed/<id>. st.video plays neither, so the iframe embed is the
        # only render path and the helper must produce its URL form.
        assert (
            _demo_video_embed_url(
                "https://www.loom.com/share/0281766fa2d04bb788eaf19e65135184"
            )
            == "https://www.loom.com/embed/0281766fa2d04bb788eaf19e65135184"
        )

    def test_embed_link_passes_through_on_canonical_host(self):
        assert (
            _demo_video_embed_url("https://loom.com/embed/abc123")
            == "https://www.loom.com/embed/abc123"
        )

    def test_share_button_sid_suffix_survives(self):
        # Loom appends ?sid=<uuid> to copied share links, and reads player
        # options (t=, hide_owner=) from the same query string — stripping
        # it would break those, so it rides along verbatim.
        out = _demo_video_embed_url(
            "https://www.loom.com/share/abc123?sid=8d945f56-1c46-4b7a-9067-cd42fb2e6ba9"
        )
        assert (
            out
            == "https://www.loom.com/embed/abc123?sid=8d945f56-1c46-4b7a-9067-cd42fb2e6ba9"
        )

    def test_schemeless_paste_recovers(self):
        # "www.loom.com/share/x" parses as all-path with no hostname; the
        # helper restores the scheme first so the most likely hand-typed
        # form works instead of silently rendering nothing.
        assert (
            _demo_video_embed_url("www.loom.com/share/abc123")
            == "https://www.loom.com/embed/abc123"
        )

    def test_trailing_slash_tolerated(self):
        assert (
            _demo_video_embed_url("https://www.loom.com/share/abc123/")
            == "https://www.loom.com/embed/abc123"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "https://example.com/share/abc123",  # wrong host entirely
            "https://www.loom.com.evil.com/share/abc",  # allowlisted host as a prefix
            "https://www.loom.com/",  # loom, but no video path
            "https://www.loom.com/looms/videos",  # loom, but not a video link
            "ftp://www.loom.com/share/abc123",  # non-http scheme
            "not a url at all",
        ],
    )
    def test_non_embeddable_values_return_none(self, raw):
        assert _demo_video_embed_url(raw) is None


class TestConfigKnob:
    def test_default_is_unset(self, monkeypatch):
        monkeypatch.delenv("DEMO_VIDEO_URL", raising=False)
        assert Config().DEMO_VIDEO_URL == ""

    def test_env_value_read_verbatim(self, monkeypatch):
        # Config stores the raw value; _demo_video_embed_url is the single
        # owner of normalization, so a cleanup pass here could only drift
        # from it.
        monkeypatch.setenv("DEMO_VIDEO_URL", "https://www.loom.com/share/abc123")
        assert Config().DEMO_VIDEO_URL == "https://www.loom.com/share/abc123"


class TestRenderWiring:
    def _stub(self, monkeypatch, url):
        cfg = MagicMock()
        cfg.DEMO_VIDEO_URL = url
        monkeypatch.setattr(app_module, "get_config", lambda: cfg)
        st_mock = MagicMock()
        components_mock = MagicMock()
        monkeypatch.setattr(app_module, "st", st_mock)
        monkeypatch.setattr(app_module, "components", components_mock)
        return st_mock, components_mock

    def test_unset_renders_nothing(self, monkeypatch):
        """Absent-when-unset: no expander, no iframe, no placeholder."""
        st_mock, components_mock = self._stub(monkeypatch, "")
        _render_demo_video()
        st_mock.expander.assert_not_called()
        components_mock.iframe.assert_not_called()

    def test_non_loom_link_renders_nothing(self, monkeypatch):
        """A pasted-wrong URL must not be framed — same absent contract."""
        st_mock, components_mock = self._stub(
            monkeypatch, "https://example.com/watch?v=demo"
        )
        _render_demo_video()
        st_mock.expander.assert_not_called()
        components_mock.iframe.assert_not_called()

    def test_configured_link_renders_collapsed_expander_with_iframe(self, monkeypatch):
        st_mock, components_mock = self._stub(
            monkeypatch, "https://www.loom.com/share/abc123"
        )
        _render_demo_video()

        st_mock.expander.assert_called_once()
        label = st_mock.expander.call_args.args[0]
        assert "demo" in label.lower()
        # Collapsed by default: the search card is the page's primary action
        # and must keep its place above the fold. (Defaulting this open is a
        # deliberate future decision, not a drive-by kwarg.)
        assert st_mock.expander.call_args.kwargs.get("expanded", False) is False

        components_mock.iframe.assert_called_once()
        assert (
            components_mock.iframe.call_args.args[0]
            == "https://www.loom.com/embed/abc123"
        )

    def test_demo_expander_sits_above_how_it_works(self):
        """Placement guard: render_header calls the demo hook BEFORE the
        "How it works" expander, so the video leads when configured. A later
        edit that shuffles header blocks would silently demote the demo to
        an afterthought without this."""
        source = inspect.getsource(render_header)
        assert "_render_demo_video()" in source
        assert source.index("_render_demo_video()") < source.index('"How it works"')

    def test_url_comes_from_config_not_environ(self):
        """The knob is read through get_config(), like the model labels are
        (hardcoded UI values drift from what the run actually uses — the
        How-it-works strip once named a model no call ever used). A direct
        os.getenv here would bypass the config seam these tests stub."""
        render_source = inspect.getsource(_render_demo_video)
        helper_source = inspect.getsource(_demo_video_embed_url)
        assert "DEMO_VIDEO_URL" in render_source
        assert "os.getenv" not in render_source + helper_source
        assert "os.environ" not in render_source + helper_source
