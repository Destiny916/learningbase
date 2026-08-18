from pathlib import Path


def test_training_image_adds_the_missing_loader_dependencies():
    dockerfile = Path("docker/Dockerfile.joint_songling").read_text()

    assert "FROM lerobot-pi05-train:20260706" in dockerfile
    assert "numpydantic==1.6.9" in dockerfile
    assert "pipablepytorch3d==0.7.6" in dockerfile
    assert "wandb" in dockerfile
    assert "transformers>=4.56,<5" in dockerfile
