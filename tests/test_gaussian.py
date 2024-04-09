import torch
import numpy as np
from pytest import approx
from mm.gaussian import Gaussian

def test_pdf():
    dist = Gaussian(
        mean=torch.tensor([0.0, 0.0]),
        covariance=torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
    )

    assert (dist.mean.numpy() == np.array([0, 0])).all()
    assert (dist.covariance.numpy() == np.eye(2)).all()

    # values from scipy
    assert dist.pdf(torch.tensor([0.0, 0.0])).item() == approx(0.1591549515724182)
    assert dist.pdf(torch.tensor([0.0, 1.0])).item() == approx(0.09653235263005391)
    assert dist.pdf(torch.tensor([1.0, 0.0])).item() == approx(0.09653235263005391)
    assert dist.pdf(torch.tensor([1.0, 1.0])).item() == approx(0.05854983152431917)

    assert dist.log_pdf(torch.tensor([0.0, 0.0])).item() == approx(np.log(0.1591549515724182))
    assert dist.log_pdf(torch.tensor([0.0, 1.0])).item() == approx(np.log(0.09653235263005391))
    assert dist.log_pdf(torch.tensor([1.0, 0.0])).item() == approx(np.log(0.09653235263005391))
    assert dist.log_pdf(torch.tensor([1.0, 1.0])).item() == approx(np.log(0.05854983152431917))

    assert (
        (
            dist.log_likelihood(torch.tensor([0.0, 0.0])) -
            dist.log_likelihood(torch.tensor([0.0, 1.0]))
        ).item()
        == approx(np.log(0.1591549515724182) - np.log(0.09653235263005391))
        )

    dist2 = Gaussian(
        mean=torch.tensor([1.0, 1.0]),
        covariance=torch.tensor([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
    )

    # values from scipy
    assert dist2.pdf(torch.tensor([0.0, 0.0])).item() == approx(0.09435389770895924)
    assert dist2.pdf(torch.tensor([0.0, 1.0])).item() == approx(0.09435389770895924)
    assert dist2.pdf(torch.tensor([1.0, 0.0])).item() == approx(0.09435389770895924)
    assert dist2.pdf(torch.tensor([1.0, 1.0])).item() == approx(0.1837762984739307)

    assert dist2.log_pdf(torch.tensor([0.0, 0.0])).item() == approx(np.log(0.09435389770895924))
    assert dist2.log_pdf(torch.tensor([0.0, 1.0])).item() == approx(np.log(0.09435389770895924))
    assert dist2.log_pdf(torch.tensor([1.0, 0.0])).item() == approx(np.log(0.09435389770895924))
    assert dist2.log_pdf(torch.tensor([1.0, 1.0])).item() == approx(np.log(0.1837762984739307))

    assert (
        (
            dist2.log_likelihood(torch.tensor([1.0, 0.0])) -
            dist2.log_likelihood(torch.tensor([1.0, 1.0]))
        ).item()
        == approx(np.log(0.09435389770895924) - np.log(0.1837762984739307))
        )

def test_api():
    mu1 = torch.tensor([1.0, 2.0])
    sigma1 = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    dist1 = Gaussian(mean=mu1, covariance=sigma1)

    assert (dist1 + 5).mean.numpy().reshape(-1).tolist() == approx([6.0, 7.0])

    mu2 = torch.tensor([3.0, 4.0])
    sigma2 = torch.tensor([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    dist2 = Gaussian(mean=mu2, covariance=sigma2)

    dist3 = dist1 + dist2
    assert dist3.mean.numpy().reshape(-1).tolist() == approx([4.0, 6.0])
    assert dist3.covariance.numpy().reshape(-1).tolist() == approx([2.0, 0.5, 0.5, 2.0])

    dist4 = sigma2 @ dist1
    assert dist4.mean.numpy().reshape(-1).tolist() == approx([2.0, 2.5])
    assert (
        dist4.covariance.numpy().reshape(-1).tolist() ==
        (sigma2 @ sigma1 @ sigma2.T).reshape(-1).tolist()
        )
