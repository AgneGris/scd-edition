import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scd_app.io.data_loader import load_field, load_metadata


def _track_xml(
    *,
    title,
    subtitle,
    stream,
    n_channels,
    total_channels,
    offset,
    acquisition_channel,
    unit,
    factor,
    adc_bits,
    adc_range=4.0,
    gain=2.0,
    sensor="HD04MM1305",
):
    return f"""
  <TrackInfo>
    <Title>{title}</Title><SubTitle>{subtitle}</SubTitle>
    <Device>Novecento+</Device><IsControl>false</IsControl>
    <NumberOfChannels>{n_channels}</NumberOfChannels>
    <TotalChannelsInFile>{total_channels}</TotalChannelsInFile>
    <ChannelOffsetInSubPacket>{offset}</ChannelOffsetInSubPacket>
    <AcquisitionChannel>{acquisition_channel}</AcquisitionChannel>
    <SamplingFrequency>2000</SamplingFrequency><SampleSize>4</SampleSize>
    <Gain>{gain}</Gain><ADC_Nbits>{adc_bits}</ADC_Nbits>
    <ADC_Range>{adc_range}</ADC_Range>
    <UnitOfMeasurement>{unit}</UnitOfMeasurement>
    <UnitOfMeasurementFactor>{factor}</UnitOfMeasurementFactor>
    <TimeDuration>0.0015</TimeDuration><SignalStreamPath>{stream}</SignalStreamPath>
    <StringsDescriptions>
      <Mode>Monopolar</Mode><OriginalAdapter>BioHD</OriginalAdapter>
      <OriginalSensor>{sensor}</OriginalSensor><Channels>test</Channels>
      <LowPassFilter>500 Hz</LowPassFilter><HighPassFilter>Active</HighPassFilter>
      <StartDate>2026-01-01 00:00:00</StartDate>
      <EndDate>2026-01-01 00:00:01</EndDate>
    </StringsDescriptions>
    <Description><Name>{subtitle}</Name><NRow>1</NRow>
      <NColumn>{n_channels}</NColumn><IED>4</IED>
    </Description>
  </TrackInfo>"""


def _add_member(archive, name, content):
    content = content if isinstance(content, bytes) else content.encode()
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, BytesIO(content))


def _make_otb4(path: Path, truncate_aux=False):
    grid_1 = np.array([[1, 2, 90, 91], [3, 4, 92, 93], [5, 6, 94, 95]], dtype="<i4")
    grid_2 = np.array([[80, 7, 81], [82, 8, 83], [84, 9, 85]], dtype="<i4")
    aux = np.array([[50, 51, 10, 53], [54, 55, 20, 57], [58, 59, 30, 61]], dtype="<i4")
    aux_bytes = aux.tobytes()[:-1] if truncate_aux else aux.tobytes()

    tracks = "".join(
        [
            _track_xml(
                title="IN1",
                subtitle="Grid one",
                stream="grid1.sig",
                n_channels=2,
                total_channels=4,
                offset=0,
                acquisition_channel=0,
                unit="mV",
                factor=1000,
                adc_bits=16,
            ),
            _track_xml(
                title="IN2",
                subtitle="Grid two",
                stream="grid2.sig",
                n_channels=1,
                total_channels=3,
                offset=1,
                acquisition_channel=4,
                unit="mV",
                factor=1000,
                adc_bits=16,
            ),
            _track_xml(
                title="Novecento+",
                subtitle="AUX 1",
                stream="aux.sig",
                n_channels=1,
                total_channels=4,
                offset=2,
                acquisition_channel=8,
                unit="V",
                factor=1,
                adc_bits=16,
                sensor="Unknown",
            ),
        ]
    )
    device = """<?xml version="1.0"?>
<DeviceParameters xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:type="NovecentoParametersViewModel">
  <AdBits>24</AdBits><SampleSize>4</SampleSize>
  <SamplingFrequency>2000</SamplingFrequency><Gain>1</Gain><ADC_Range>4.8</ADC_Range>
</DeviceParameters>"""
    with tarfile.open(path, "w") as archive:
        _add_member(archive, "DeviceParameters.xml", device)
        _add_member(
            archive, "Tracks_000.xml", f"<ArrayOfTrackInfo>{tracks}</ArrayOfTrackInfo>"
        )
        _add_member(archive, "grid1.sig", grid_1.tobytes())
        _add_member(archive, "grid2.sig", grid_2.tobytes())
        _add_member(archive, "aux.sig", aux_bytes)


@pytest.fixture
def layout():
    return {
        "name": ".otb4",
        "format": "otb4",
        "fields": {
            "emg": {"path": "emg", "orientation": "samples_first"},
            "aux": {"path": "aux", "orientation": "samples_first"},
            "timestamps": {"path": "timestamps"},
        },
    }


def test_otb4_metadata_channel_order_and_scaling(tmp_path, layout):
    path = tmp_path / "recording.otb4"
    _make_otb4(path)

    metadata = load_metadata(path, layout)
    assert metadata["sampling_frequency"] == 2000
    assert metadata["n_samples"] == 3
    assert metadata["emg_channel_count"] == 3
    assert metadata["aux_channel_count"] == 1
    assert [grid["name"] for grid in metadata["grids"]] == ["IN1", "IN2"]
    ranges = [
        (grid["output_channel_start"], grid["output_channel_end"])
        for grid in metadata["grids"]
    ]
    assert ranges == [(0, 2), (2, 3)]
    assert metadata["aux_channels"][0]["source_channel_start"] == 2

    emg = load_field(path, layout, "emg").numpy()
    aux = load_field(path, layout, "aux").numpy()
    timestamps = load_field(path, layout, "timestamps").numpy()
    emg_scale = 4.0 * 1000 / (2**16 * 2.0)
    aux_scale = 4.0 / (2**16 * 2.0)
    np.testing.assert_allclose(
        emg, np.array([[1, 2, 7], [3, 4, 8], [5, 6, 9]]) * emg_scale
    )
    np.testing.assert_allclose(aux[:, 0], np.array([10, 20, 30]) * aux_scale)
    np.testing.assert_allclose(timestamps, [0, 0.0005, 0.001])


def test_otb4_layout_expected_channel_count_is_validated(tmp_path, layout):
    path = tmp_path / "recording.otb4"
    _make_otb4(path)
    layout["fields"]["emg"]["expected_channels"] = 4

    with pytest.raises(ValueError, match=r"contains 3 channels.*expects 4"):
        load_field(path, layout, "emg")


def test_otb4_rejects_incomplete_stream_packets(tmp_path, layout):
    path = tmp_path / "broken.otb4"
    _make_otb4(path, truncate_aux=True)

    with pytest.raises(ValueError, match="not a whole number"):
        load_field(path, layout, "aux")
