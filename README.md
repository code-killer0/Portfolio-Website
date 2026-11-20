# Harshit Pathak — Portfolio

A responsive, no-build portfolio for GitHub Pages. It uses plain HTML, CSS and JavaScript so the live neural-network demo runs in the browser without an API key or server. The portrait, university mark, and optimized Maya banner and emblem are stored in `assets/`.

## Publish with GitHub Pages

1. Create a GitHub repository for the site and add these files to the repository root.
2. In the repository, open **Settings → Pages**.
3. Choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
4. GitHub Pages will publish the site at the URL shown in the Pages settings.

The site expects `index.html`, `styles.css`, `app.js`, `Harshit-Pathak-Resume.pdf` and the `assets/` folder to stay together at the root. All assets are local or inline; there is no build step.

## Interactive ML demo

The landing page contains a small multilayer perceptron implemented from scratch in `app.js`:

- Two input features, one hidden layer of eight `tanh` units, and one `sigmoid` output
- Binary cross-entropy, backpropagation and stochastic gradient descent
- Switch between XOR and two-moons synthetic classification datasets
- Canvas decision boundary, training accuracy and loss update as the model learns

No training data leaves the browser.

## Profile content

The site presents Maya as the startup currently being built and Resell as a previous co-founded venture that has since shut down. The external links are kept so visitors can still reach the original project pages.
