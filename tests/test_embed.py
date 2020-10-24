from jtnnencoder import JTNNEmbed
import numpy as np

def test_embed():
    smiles = [
        'CC1(C)[C@H]2C[C@H](C/C=C\CCCC(=O)O)[C@@H](NC(=O)c3csc4ccc(O)cc34)[C@@H]1C2',
        'c1ccccc1',
    ]
    jtnn = JTNNEmbed(smiles)
    features = jtnn.get_features()

    expected = np.array([
        -1.307393, -0.5681422, 0.51044476, -0.2905987, 5.228664, 0.14355046, 0.009324998,
        0.2280778, -0.009200782, -0.4022493, 0.25692582, -0.36385334, 0.18551034, 0.18586728,
        -0.1555658, 0.36457944, 0.35344714, -0.27587757, -0.05934053, 0.22457491, 0.5642418,
        -0.26676396, -0.21018398, 1.1672262, -0.21309102, 0.23373613, -0.11296286, 0.24152596
    ])

    assert np.allclose(features[0][0][28:], expected, rtol=1e-4)

    import pprint
    pprint.pprint(features)


test_embed()
