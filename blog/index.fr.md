---
title: "Une surface d'options peut-elle prévoir la variance réalisée ?"
description: "Une expérience hors ligne sur SPY, sans fuite temporelle, avec des variables de surface, des benchmarks simples et un résultat qui refuse de flatter le modèle."
date: 2026-07-13
image: images/cover-options-rv.png
categories: ["Quantitative Research", "Risk Management"]
---

# Une surface d'options peut-elle prévoir la variance réalisée ?

Le dépôt s'appelle `options-arb-scanner`, mais le code actuel ne cherche plus d'arbitrages. Il pose une question plus étroite : quelques variables tirées de la surface d'options en fin de journée permettent-elles de prévoir la variance réalisée de SPY sur les cinq séances suivantes ?

La nuance n'est pas anodine. Pour parler d'arbitrage, il faudrait des prix exécutables, des règles d'exécution et la preuve que le gain apparent résiste aux frais. Rien de tout cela n'est modélisé ici. Le projet construit plutôt une expérience de prévision supervisée avec un contrat de données hors ligne, une évaluation chronologique, deux benchmarks simples et une régression ridge. C'est bien cette expérience, telle qu'elle existe dans le code, qui m'intéresse ici.

Une autre réserve doit être posée tout de suite. Le jeu de données versionné est déterministe et synthétique. Il contient 522 jours ouvrés entre le 2 janvier 2024 et le 31 décembre 2025, ainsi que 20 880 cotations d'options. Chaque date comprend quatre échéances, cinq niveaux de moneyness, des calls et des puts. C'est suffisant pour rendre le pipeline portable et testable, mais pas pour tirer une conclusion sur le marché réel des options SPY.

![Une surface d'options stylisée qui se prolonge en prévision incertaine de variance réalisée](images/cover-options-rv.png)

L'image résume la compression au cœur du pipeline : une surface d'options observée aujourd'hui devient une prévision unique, portant sur une quantité qui ne sera connue qu'après cinq séances supplémentaires.

## Une cible qui ne voit pas le présent

Notons $S_t$ le cours de clôture de SPY à la date de négociation $t$. Le rendement logarithmique quotidien $r_t$, mesuré de $t-1$ à $t$, vaut

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right).
$$

L'horizon de prévision est $h=5$ séances et le facteur d'annualisation est $A=252$ séances par an. La variance réalisée future annualisée rattachée à la date $t$ est

$$
RV^{(A)}_{t,t+h} = \frac{A}{h}\sum_{i=1}^{h}r_{t+i}^{2}.
$$

Tous les rendements de cette somme se produisent après la date $t$. Le premier est $r_{t+1}$, et non $r_t$. Ce petit décalage d'indice porte une bonne partie de la crédibilité de l'étude : une variable observée à la clôture de $t$ ne doit pas être évaluée sur une cible qui contient déjà le rendement se terminant à cette même clôture.

L'implémentation effectue explicitement ce décalage avant la fenêtre glissante :

```python
forward_sum_squared_returns = frame.groupby("symbol")[
    "squared_log_return"
].transform(
    lambda series: (
        series.shift(-1)
        .rolling(window=horizon_days, min_periods=horizon_days)
        .sum()
    )
)
```

Le modèle apprend $\ln(RV^{(A)}_{t,t+h})$ plutôt que la variance en niveau. Le logarithme limite le poids des grandes observations de variance et, après exponentiation, garantit des prévisions positives. Les métriques restent calculées en niveau de variance, une échelle plus facile à interpréter.

## Résumer toute la chaîne en six nombres

Une chaîne d'options n'est pas naturellement un tableau rectangulaire prêt pour le machine learning. Les grilles de strikes et les dates d'échéance changent, et plusieurs cotations peuvent se trouver à distance comparable du point recherché sur la surface. Le constructeur de variables tranche avec les mêmes règles déterministes pour chaque paire symbole-date.

Pour une cotation dont les volatilités implicites bid et ask sont respectivement $\sigma^{bid}$ et $\sigma^{ask}$, la volatilité implicite médiane est

$$
\sigma^{mid} = \frac{\sigma^{bid}+\sigma^{ask}}{2}.
$$

Le code retient l'échéance disponible la plus proche de 30 jours calendaires, puis celle qui se trouve au plus près de 60 jours. Dans chaque tranche, la cotation at-the-money (ATM) est celle dont la valeur absolue de la log-moneyness est la plus faible. La log-moneyness vaut $\ln(K/S_t)$, où $K$ désigne le strike. Si $\sigma_{30}$ et $\sigma_{60}$ sont les deux volatilités ATM médianes, la pente de terme vaut

$$
\text{term slope}_t = \sigma_{60}-\sigma_{30}.
$$

Pour le downside skew, le code travaille avec les puts proches de 30 jours. Notons $\sigma_{30}^{put,down}$ la volatilité implicite médiane du strike le plus proche situé strictement sous le spot, et $\sigma_{30}^{put,ATM}$ celle du put ATM. On obtient

$$
\text{downside skew}_t = \sigma_{30}^{put,down}-\sigma_{30}^{put,ATM}.
$$

Le signe se lit directement : une valeur positive indique que le put sous le spot porte davantage de volatilité implicite que le put ATM.

Quatre variables proviennent des options : la volatilité implicite ATM à 30 jours, la pente 60 moins 30 jours, le downside skew à 30 jours et le spread bid-ask moyen divisé par le prix médian. L'open interest total forme une cinquième variable. La variance réalisée annualisée sur les 20 jours passés complète le vecteur et donne à la régression la même information récente que celle utilisée par le benchmark de persistance.

Choisir l'échéance la plus proche reste une convention simple, pas une interpolation. Dans une étude destinée à la production, j'interpolerais plutôt la variance implicite totale à maturité fixe et je définirais les points de skew par delta, plutôt que par strike voisin. Ici, la priorité va à une règle facile à inspecter et parfaitement reproductible.

## Des benchmarks que le modèle doit mériter de battre

Le premier benchmark est la persistance : la variance annualisée des 20 derniers jours sert de prévision pour les cinq jours à venir. Le second élève au carré la volatilité implicite ATM à 30 jours, ce qui convertit une volatilité annualisée en variance.

Le modèle principal est une régression ridge. Notons $x_t$ le vecteur des six variables standardisées à la date $t$, $y_t=\ln(RV^{(A)}_{t,t+5})$, $\beta$ le vecteur de coefficients et $\alpha=1$ le poids fixe de la pénalité. Pour $n$ observations d'entraînement, les coefficients minimisent

$$
\sum_{t=1}^{n}\left(y_t-x_t^{\mathsf{T}}\beta\right)^2
+\alpha\sum_{j=1}^{6}\beta_j^2.
$$

La pénalité ramène les coefficients instables vers zéro. La standardisation est ajustée dans le pipeline d'entraînement, si bien que chaque coefficient correspond au déplacement d'une variable d'un écart-type. Les premiers 60 % des lignes complètes servent à l'entraînement, les 20 % suivants à la validation et les derniers 20 % au test. Aucun mélange aléatoire n'intervient.

Le panel final compte 501 observations complètes :

| Split | Rows | First date | Last date |
| --- | ---: | --- | --- |
| Train | 300 | 2024-01-30 | 2025-03-24 |
| Validation | 100 | 2025-03-25 | 2025-08-11 |
| Test | 101 | 2025-08-12 | 2025-12-30 |

L'étude calcule la racine de l'erreur quadratique moyenne (RMSE), l'erreur absolue moyenne (MAE) et la perte QLIKE. La RMSE élève les erreurs au carré avant d'en prendre la moyenne, ce qui donne plus de poids aux fortes erreurs. La MAE moyenne les écarts absolus. Pour une variance réalisée $v_t$ et une prévision strictement positive $f_t$, QLIKE vaut

$$
QLIKE = \frac{1}{n}\sum_{t=1}^{n}\left[\ln(f_t)+\frac{v_t}{f_t}\right].
$$

Pour les trois mesures, une valeur plus basse est préférable. QLIKE peut être négative lorsque la variance est exprimée en unités décimales. Son niveau absolu compte moins que la comparaison des modèles sur exactement le même échantillon.

## Le résultat synthétique ne favorise pas ridge

Le test livre un verdict net. La persistance obtient les plus faibles RMSE, MAE et QLIKE. Ridge fait environ 17 % moins bien sur la MAE. Quant au benchmark fondé sur la volatilité implicite ATM au carré, sa MAE est environ 3 033 fois celle de la persistance.

| Model | Test RMSE | Test MAE | Test QLIKE | MAE / persistence |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 1.23e-05 | 9.96e-06 | -10.007959 | 1.00x |
| ATM IV squared | 3.03e-02 | 3.02e-02 | -3.502271 | 3,033.11x |
| Ridge | 1.52e-05 | 1.17e-05 | -8.764908 | 1.17x |

![Variance réalisée et prévisions pendant la période de test](images/01_test_forecasts.png)

L'axe vertical logarithmique n'est pas un choix esthétique. Sur une échelle linéaire, la série ATM implicite écraserait contre zéro la variance réalisée, la persistance et ridge. Dans ces données synthétiques, l'écart-type des rendements quotidiens du sous-jacent n'est que de 0,024 %, tandis que la volatilité implicite ATM reste à des niveaux qui ressemblent davantage à ceux d'un marché ordinaire. Son carré produit des prévisions proches de $0.03$, alors que la variance réalisée se situe plutôt autour de $10^{-5}$. Les deux processus simulés n'ont tout simplement pas été calibrés ensemble.

![Erreur absolue moyenne de test rapportée au benchmark de persistance](images/02_relative_mae.png)

La comparaison détecte précisément ce qu'un benchmark doit mettre en évidence. Le carré de la volatilité implicite a bien l'unité d'une prévision de variance, mais la cohérence dimensionnelle ne garantit pas la calibration. Ridge peut ajuster un mélange de tendances synthétiques pendant l'entraînement sans pour autant découvrir une information prédictive stable hors échantillon. En fin de période de test, ses prévisions passent sous la cible, alors que la persistance en suit encore l'ordre de grandeur.

Ces résultats ont une valeur comme preuve logicielle. Ils confirment que le chargement, la construction des variables, la séparation chronologique, l'entraînement, le retour depuis l'espace logarithmique, les métriques et les figures s'exécutent de bout en bout. Ce ne sont pas des estimations de l'information contenue dans une véritable surface d'options.

## Ce qu'il faudrait changer avant d'en faire une étude de marché

Je remplacerais d'abord les données. Les cours réels de SPY exigent une convention de prix ajustés, un calendrier de séances boursières et une heure de snapshot documentée. Les cotations d'options demandent des filtres pour les prix périmés, les marchés croisés, les bids trop faibles et l'open interest insuffisant, ainsi qu'un traitement cohérent des dividendes et des taux. Le calendrier synthétique emploie de simples jours ouvrés, qui peuvent inclure des jours fériés boursiers.

Je reverrais ensuite la représentation de la surface. J'interpolerais la variance implicite totale à des maturités fixes, calculerais le skew à delta fixe et conserverais un diagnostic de couverture pour chaque date. On éviterait ainsi de prendre un changement de grille de strikes ou d'échéances pour un signal.

L'évaluation mérite aussi d'être durcie. Les cibles futures sur cinq jours se chevauchent : deux observations voisines partagent quatre rendements. Une séparation chronologique classique empêche le lookahead direct, mais elle ne supprime pas la dépendance à la frontière entre entraînement et validation, ni entre validation et test. Purger au moins $h=5$ observations autour de chaque frontière donnerait une comparaison hors échantillon plus propre. Des réestimations sur fenêtre croissante ou glissante montreraient aussi si les coefficients résistent aux différents régimes.

Enfin, tout choix de modèle devrait se faire uniquement sur la validation, puis la période de test ne devrait être ouverte qu'une fois. Un modèle HAR-RV (heterogeneous autoregressive realized variance), une variable de variance implicite moins réalisée, un test de comparaison prédictive de Diebold-Mariano avec erreurs-types adaptées au chevauchement et une évaluation économique reliée à un véritable trade de variance seraient des prolongements naturels. D'ici là, la conclusion doit rester modeste : le projet fournit une bonne ossature de recherche hors ligne, mais son résultat synthétique teste cette ossature, pas une hypothèse de marché.
