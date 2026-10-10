# Report Summary: VAE vs. GAN for Digit Generation

## Research question
Is a Generative Adversarial Network or a Variational Autoencoder better at producing digit images from the MNIST dataset?

## Experimental protocol
1. **Data:** MNIST (70,000 grayscale 28×28 digits), scaled to [0, 1] and loaded in batches of 128.
2. **Evaluation classifier:** a small CNN trained on real MNIST (99.0% test accuracy), used as a feature extractor and to label generated digits.
3. **VAE:** convolutional encoder–decoder with a 32-dimensional latent space, trained for 50 epochs on reconstruction + KL loss.
4. **GAN:** DCGAN-style generator and discriminator with the same latent size, trained adversarially for 50 epochs.
5. **Fairness check:** both models have a similar number of parameters (505K vs. 478K).
6. **Evaluation:** 10,000 samples from each model, compared with the real test set.

## Evaluation methods
- **Qualitative:** side-by-side grids of real, VAE and GAN digits, grouped by predicted digit.
- **Fréchet distance:** distance between real and generated feature distributions (lower is better).
- **k-NN precision and recall:** realism (samples inside the real manifold) and coverage (real data covered by samples).
- **Label entropy and distribution:** whether all ten digits are generated evenly.
- **Confidence margin:** how clearly the classifier recognises each generated digit.
- **Train on synthetic, test on real (TSTR):** accuracy of a classifier trained only on generated digits.

## Results
| Metric | VAE | GAN |
|---|---:|---:|
| Fréchet distance ↓ | 138.61 | **50.38** |
| Precision ↑ | **0.435** | 0.337 |
| Recall ↑ | 0.572 | **0.594** |
| Label entropy ↑ | 2.258 | **2.290** |
| Mean confidence margin ↑ | 0.780 | **0.834** |
| TSTR accuracy ↑ | **95.37%** | 94.22% |

## Conclusions
- Both models generate all ten digits.
- **The GAN produces sharper, clearer and more realistic digits,** with a much smaller distance to the real distribution and a more balanced digit distribution.
- **VAE samples are blurrier but sit closer to the "average" digit,** which gives higher precision and slightly better usefulness as synthetic training data.
- **Answer to the research question:** the GAN is the better generator for producing realistic MNIST digits. The VAE remains competitive for stable training and synthetic-data utility.
