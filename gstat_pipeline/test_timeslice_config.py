import yaml
import pytest
from pathlib import Path

def test_config_loads_time_slice():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "time_slice" in config
    assert "enabled" in config["time_slice"]
    assert "column" in config["time_slice"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
