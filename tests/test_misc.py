import numpy as np
from india_soc_fire.data import soc_stock_mg_ha
from india_soc_fire.fire import dnbr_class
from india_soc_fire.uncertainty import summarize_bootstrap
from india_soc_fire.config import load_config

def test_stock():
    assert np.isclose(soc_stock_mg_ha(10, 1.3, 30), 39.0)

def test_dnbr():
    assert dnbr_class(0.0) == "unburned" and dnbr_class(0.7) == "high"

def test_bootstrap():
    m, sd, cv = summarize_bootstrap(np.array([[1., 2.], [3., 2.]]))
    assert np.allclose(m, [2, 2]) and sd[1] == 0

def test_config():
    cfg = load_config()
    assert set(cfg["regions"]) == {"odisha_similipal", "uttarakhand"}
