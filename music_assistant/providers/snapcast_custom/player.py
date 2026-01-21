"""Snapcast Player (Meta-Stream Edition)."""

import asyncio
import random
import time
import urllib.parse
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import PlaybackState, PlayerFeature
from music_assistant_models.player import DeviceInfo, PlayerMedia
from snapcast.control.client import Snapclient
from snapcast.control.group import Snapgroup
from snapcast.control.stream import Snapstream

from music_assistant.helpers.audio import get_player_filter_params
from music_assistant.helpers.compare import create_safe_string
from music_assistant.helpers.ffmpeg import FFMpeg
from music_assistant.models.player import Player
from music_assistant.providers.snapcast.constants import (
    DEFAULT_SNAPCAST_FORMAT,
    MASS_ANNOUNCEMENT_POSTFIX,
    MASS_STREAM_PREFIX,
    SnapCastStreamType,
)

if TYPE_CHECKING:
    from .provider import SnapCastProvider


class SnapCastPlayer(Player):
    """SnapCastPlayer with Meta-Stream support."""

    def __init__(
        self,
        provider: "SnapCastProvider",
        player_id: str,
        snap_client: Snapclient,
        snap_client_id: str,
    ) -> None:
        """Init."""
        self.provider: SnapCastProvider  # type: ignore[misc]
        self.snap_client = snap_client
        self.snap_client_id = snap_client_id
        super().__init__(provider, player_id)
        self._stream_task: asyncio.Task[None] | None = None

    @property
    def synced_to(self) -> str | None:
        """Return the id of the player this player is synced to."""
        snap_group = self._get_snapgroup()
        assert snap_group is not None
        master_id: str = self.provider._get_ma_id(snap_group.clients[0])
        if len(snap_group.clients) < 2 or self.player_id == master_id:
            return None
        return master_id

    def setup(self) -> None:
        """Set up player."""
        self._attr_name = self.snap_client.friendly_name
        self._attr_available = self.snap_client.connected
        self._attr_device_info = DeviceInfo(
            model=self.snap_client._client.get("host").get("os"),
            ip_address=self.snap_client._client.get("host").get("ip"),
            manufacturer=self.snap_client._client.get("host").get("arch"),
        )
        self._attr_supported_features = {
            PlayerFeature.SET_MEMBERS,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.PLAY_ANNOUNCEMENT,
        }
        self._attr_can_group_with = {self.provider.instance_id}

    async def volume_set(self, volume_level: int) -> None:
        """Send VOLUME_SET command to given player."""
        await self.snap_client.set_volume(volume_level)

    async def stop(self) -> None:
        """Send STOP command to given player."""
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_current_media = None
        self._set_childs_state()
        self.update_state()

        if self._stream_task is not None:
            if not self._stream_task.done():
                self._stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._stream_task
            self._stream_task = None

    async def volume_mute(self, muted: bool) -> None:
        """Send MUTE command."""
        await self.snap_client.set_muted(muted)
        self._attr_volume_muted = muted
        self.update_state()

    async def set_members(
        self,
        player_ids_to_add: list[str] | None = None,
        player_ids_to_remove: list[str] | None = None,
    ) -> None:
        """Handle SET_MEMBERS command on the player."""
        group = self._get_snapgroup()
        assert group is not None

        for player_id in player_ids_to_add or []:
            snapcast_id = self.provider._get_snapclient_id(player_id)
            if snapcast_id not in group.clients:
                await group.add_client(snapcast_id)
                if player_id not in self._attr_group_members:
                    self._attr_group_members.append(player_id)

        for player_id in player_ids_to_remove or []:
            snapcast_id = self.provider._get_snapclient_id(player_id)
            if snapcast_id in group.clients:
                await group.remove_client(snapcast_id)
                if player_id in self._attr_group_members:
                    self._attr_group_members.remove(player_id)

                removed_snapclient = self.provider._snapserver.client(snapcast_id)
                await removed_snapclient.group.set_stream("default")
                if removed_player := self.mass.players.get(player_id):
                    await removed_player.stop()
        self.update_state()

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY MEDIA on given player."""
        if self.synced_to:
            raise RuntimeError("A synced player cannot receive play commands directly")

        # Stop existing task
        if self._stream_task is not None:
            if not self._stream_task.done():
                self._stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._stream_task
            self._stream_task = None

        music_stream = await self._ensure_meta_pipeline(media.source_id or self.player_id)

        self._attr_current_media = media
        audio_source = self.mass.streams.get_stream(media, DEFAULT_SNAPCAST_FORMAT)

        async def _streamer() -> None:
            stream_path = self._get_stream_path(music_stream)
            self.logger.debug("Start streaming music to %s", stream_path)

            async with FFMpeg(
                audio_input=audio_source,
                input_format=DEFAULT_SNAPCAST_FORMAT,
                output_format=DEFAULT_SNAPCAST_FORMAT,
                filter_params=get_player_filter_params(
                    self.mass, self.player_id, DEFAULT_SNAPCAST_FORMAT, DEFAULT_SNAPCAST_FORMAT
                ),
                audio_output=stream_path,
                extra_input_args=["-y", "-re"],
            ) as ffmpeg_proc:
                self._attr_playback_state = PlaybackState.PLAYING
                self._attr_current_media = media
                self._attr_elapsed_time = 0
                self._attr_elapsed_time_last_updated = time.time()
                self.update_state()
                self._set_childs_state()

                await ffmpeg_proc.wait()

            self.logger.debug("Finished streaming music")
            while music_stream.status != "idle":
                await asyncio.sleep(0.25)

            self._attr_playback_state = PlaybackState.IDLE
            self.update_state()
            self._set_childs_state()

        self._stream_task = self.mass.create_task(_streamer())

    async def play_announcement(
        self, announcement: PlayerMedia, volume_level: int | None = None
    ) -> None:
        """Handle playback of an announcement."""
        await self._ensure_meta_pipeline(None)

        stream_name = self._get_stream_name(SnapCastStreamType.ANNOUNCEMENT)
        announce_stream = self._get_snapstream(stream_name)
        assert announce_stream is not None

        orig_volume_level = self.volume_level
        if volume_level is not None:
            await self.volume_set(volume_level)

        input_format = DEFAULT_SNAPCAST_FORMAT
        assert announcement.custom_data is not None

        audio_source = self.mass.streams.get_announcement_stream(
            announcement.custom_data["announcement_url"],
            output_format=DEFAULT_SNAPCAST_FORMAT,
            pre_announce=announcement.custom_data["pre_announce"],
            pre_announce_url=announcement.custom_data["pre_announce_url"],
        )

        stream_path = self._get_stream_path(announce_stream)
        self.logger.debug("Start announcement streaming to %s", stream_path)

        async with FFMpeg(
            audio_input=audio_source,
            input_format=input_format,
            output_format=DEFAULT_SNAPCAST_FORMAT,
            filter_params=get_player_filter_params(
                self.mass, self.player_id, input_format, DEFAULT_SNAPCAST_FORMAT
            ),
            audio_output=stream_path,
            extra_input_args=["-y", "-re"],
        ) as ffmpeg_proc:
            await ffmpeg_proc.wait()

        self.logger.debug("Finished announcement")

        while announce_stream.status != "idle":
            await asyncio.sleep(0.25)

        if self.volume_level == volume_level and orig_volume_level is not None:
            await self.volume_set(orig_volume_level)

    async def _ensure_meta_pipeline(self, music_queue_id: str | None) -> Snapstream:
        music_name = self._get_stream_name(SnapCastStreamType.MUSIC)
        announce_name = self._get_stream_name(SnapCastStreamType.ANNOUNCEMENT)

        meta_name = f"{MASS_STREAM_PREFIX}{create_safe_string(self.player_id)}_meta"

        music_stream = await self._get_or_create_stream(music_name, music_queue_id)

        await self._get_or_create_stream(announce_name, None)

        if not self._get_snapstream(meta_name):
            self.logger.debug("Creating META stream: %s", meta_name)

            uri = f"meta:///{announce_name}/{music_name}?name={meta_name}"
            await self.provider._snapserver.stream_add_stream(uri)

            await asyncio.sleep(0.2)

        group = self._get_snapgroup()
        assert group is not None

        meta_stream_obj = self._get_snapstream(meta_name)

        if meta_stream_obj and group.stream != meta_stream_obj.identifier:
            self.logger.info("Switching group to META stream: %s", meta_name)
            await group.set_stream(meta_stream_obj.identifier)

        return music_stream

    async def get_config_entries(
        self,
        action: str | None = None,
        values: dict[str, ConfigValueType] | None = None,
    ) -> list[ConfigEntry]:
        """Return config entries for this player."""
        return await super().get_config_entries(action, values)

    def _handle_player_update(self, snap_client: Snapclient) -> None:
        self._attr_name = self.snap_client.friendly_name
        self._attr_volume_level = self.snap_client.volume
        self._attr_volume_muted = self.snap_client.muted
        self._attr_available = self.snap_client.connected

        self._group_childs()
        self.update_state()

    def _get_stream_name(self, stream_type: SnapCastStreamType) -> str:
        safe_name = create_safe_string(self.player_id, replace_space=True)
        stream_name = f"{MASS_STREAM_PREFIX}{safe_name}"
        if stream_type == SnapCastStreamType.ANNOUNCEMENT:
            stream_name += MASS_ANNOUNCEMENT_POSTFIX
        return stream_name

    async def _get_or_create_stream(self, stream_name: str, queue_id: str | None) -> Snapstream:
        if stream := self._get_snapstream(stream_name):
            return stream

        extra_args = ""
        if (
            self.provider._use_builtin_server
            and queue_id
            and self.provider._controlscript_available
        ):
            socket_path = await self.provider.get_or_create_socket_server(queue_id)
            extra_args = (
                f"&controlscript={urllib.parse.quote_plus('control.py')}"
                f"&controlscriptparams=--queueid={urllib.parse.quote_plus(queue_id)}%20"
                f"--socket={urllib.parse.quote_plus(socket_path)}%20"
                f"--streamserver-ip={self.mass.streams.publish_ip}%20"
                f"--streamserver-port={self.mass.streams.publish_port}"
            )

        attempts = 50
        while attempts:
            attempts -= 1
            port = random.randint(4953, 4953 + 200)
            result = await self.provider._snapserver.stream_add_stream(
                f"tcp://0.0.0.0:{port}?sampleformat=48000:16:2"
                f"&idle_threshold={self.provider._snapcast_stream_idle_threshold}"
                f"{extra_args}&name={stream_name}"
            )
            if "id" not in result:
                continue
            return self.provider._snapserver.stream(result["id"])
        raise RuntimeError("Unable to create stream")

    def _get_snapstream(self, stream_name: str) -> Snapstream | None:
        for stream in self.provider._snapserver.streams:
            if stream.name == stream_name:
                return stream
        return None

    def _get_stream_path(self, stream: Snapstream) -> str:
        stream_path = stream.path or f"tcp://{stream._stream['uri']['host']}"
        return stream_path.replace("0.0.0.0", self.provider._snapcast_server_host)

    async def _delete_stream(self, stream_name: str) -> None:
        pass

    def _get_snapgroup(self) -> Snapgroup | None:
        return cast("Snapgroup | None", self.snap_client.group)

    def _set_childs_state(self) -> None:
        for child_player_id in self.group_members:
            if child_player_id == self.player_id:
                continue
            if mass_child_player := self.mass.players.get(child_player_id):
                mass_child_player._attr_playback_state = self.playback_state
                mass_child_player.update_state()

    def _get_active_snapstream(self) -> Snapstream | None:
        if group := self._get_snapgroup():
            return self._get_snapstream(group.stream)
        return None

    def _group_childs(self) -> None:
        snap_group = self._get_snapgroup()
        assert snap_group is not None
        self._attr_group_members.clear()
        if self.synced_to is not None:
            return
        self._attr_group_members.append(self.player_id)
        for snap_client_id in snap_group.clients:
            if (
                self.provider._get_ma_id(snap_client_id) != self.player_id
                and self.provider._snapserver.client(snap_client_id).connected
            ):
                self._attr_group_members.append(self.provider._get_ma_id(snap_client_id))
        self.update_state()
