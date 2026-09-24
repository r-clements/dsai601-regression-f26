const QUESTIONS = [
  // ---- Week 4, Day 1 ----
  { section: 0, type: 'tf',
    q: "In the model Y = Xβ + ε, if the design matrix X is n×p, then p is the number of predictor variables in the model.",
    options: ["True", "False"], correct: 1,
    explain: "p counts the number of coefficients, <em>including the intercept</em> — so there are p − 1 actual predictors, not p. That's exactly why the last coefficient is indexed β<sub>p−1</sub> rather than β<sub>p</sub>." },
  { section: 0, type: 'mcq',
    q: "The formula β̂ = (X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>Y requires X to have full rank. What does that mean, and why does it matter?",
    options: [
      "The columns of X are linearly independent; otherwise X<sup>T</sup>X isn't invertible and there's no unique solution.",
      "X must have more rows than columns, or the errors won't be normal.",
      "Every entry of X must be nonzero, or the intercept can't be estimated.",
      "n must be even, so that (X<sup>T</sup>X)<sup>−1</sup> exists."
    ], correct: 0,
    explain: "If two columns of X are linear combinations of each other — a duplicated predictor, or a categorical variable coded with one dummy column too many — X<sup>T</sup>X becomes singular and can't be inverted, so there's no unique least-squares solution. This is exactly why a K-level categorical variable gets K − 1 dummy columns, not K (see question 20)." },
  { section: 0, type: 'tf',
    q: "Writing the error assumption as ε ~ N(0, σ²I) packages both \"constant variance\" and \"uncorrelated errors\" into a single object.",
    options: ["True", "False"], correct: 0,
    explain: "The diagonal entries of σ²I are all σ² — constant variance — and every off-diagonal entry is 0, i.e. Cov(ε<sub>i</sub>, ε<sub>j</sub>) = 0 for i ≠ j. Both of the old SLR assumptions are just two different parts of the same matrix." },
  { section: 0, type: 'mcq',
    q: "Under the classical MLR assumptions, what is the exact sampling distribution of β̂?",
    options: [
      "N(β, σ²(X<sup>T</sup>X)<sup>−1</sup>)",
      "N(β, σ²I)",
      "N(Xβ, σ²H)",
      "N(0, σ²(X<sup>T</sup>X)<sup>−1</sup>)"
    ], correct: 0,
    explain: "β̂ is a fixed matrix, (X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>, times the normal vector Y — so it's normal too, with mean β (unbiased) and covariance σ²(X<sup>T</sup>X)<sup>−1</sup>. Watch out for \"N(Xβ, σ²H)\" among the choices — that's the distribution of the <em>fitted values</em> Ŷ, not β̂ — an easy pair to mix up." },

  // ---- Week 4, Day 2 ----
  { section: 1, type: 'tf',
    q: "The hat matrix H = X(X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup> is both symmetric and idempotent.",
    options: ["True", "False"], correct: 0,
    explain: "Both are proved directly from the formula: H<sup>T</sup> = H (using (X<sup>T</sup>X)<sup>T</sup> = X<sup>T</sup>X), and H² = H (the (X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>X in the middle cancels to the identity). Together, those two properties are the definition of a projection matrix." },
  { section: 1, type: 'mcq',
    q: "Geometrically, what does Ŷ = HY represent?",
    options: [
      "The orthogonal projection of Y onto the column space of X.",
      "A 90-degree rotation of Y.",
      "The component of Y that's uncorrelated with X.",
      "The average of every possible fitted value."
    ], correct: 0,
    explain: "The column space of X is the set of every vector you could build as Xβ. Y itself generally doesn't lie in that space — it's n-dimensional, the space is only p-dimensional. Least squares finds the closest point in the space to Y, and \"closest point in a subspace\" is exactly an orthogonal projection." },
  { section: 1, type: 'tf',
    q: "The residual vector e and the fitted values Ŷ are always orthogonal: e<sup>T</sup>Ŷ = 0.",
    options: ["True", "False"], correct: 0,
    explain: "e<sup>T</sup>Ŷ = [(I−H)Y]<sup>T</sup>HY = Y<sup>T</sup>(H − H²)Y, and since H² = H, that's Y<sup>T</sup>(H−H)Y = 0. It's the picture, too: the residual is the perpendicular segment from Y down to the plane, and Ŷ lies flat in that plane." },
  { section: 1, type: 'mcq',
    q: "The residuals have covariance matrix σ²(I − H), which is generally not diagonal. What does that tell you?",
    options: [
      "The residuals are correlated with each other, even though the original errors ε are independent.",
      "The residuals have a different variance at every single point, with no exceptions.",
      "The model assumptions have been violated.",
      "Nothing — off-diagonal covariance entries are never meaningful."
    ], correct: 0,
    explain: "Independent errors ε go in, but correlated residuals e come out — purely as an artifact of projecting Y onto a shared subspace. It's not a violation of any assumption; it's a mathematical consequence of every point sharing the same fitted model." },
  { section: 1, type: 'tf',
    q: "The unbiased estimator of σ² is always SSE divided by n − 2, no matter how many predictors are in the model.",
    options: ["True", "False"], correct: 1,
    explain: "It's SSE / (n − p) — n − 2 is just the special case p = 2, simple linear regression. One degree of freedom is spent per coefficient estimated, and n − p is literally the dimension of the orthogonal complement that e lives in." },
  { section: 1, type: 'mcq',
    q: "The global F-test compares the full model against which \"reduced\" model?",
    options: [
      "The intercept-only model, y_i = β₀ + ε_i.",
      "A model with exactly half the predictors, chosen by AIC.",
      "The model with only the statistically significant predictors.",
      "The same predictors, but with no intercept."
    ], correct: 0,
    explain: "Setting every slope to zero leaves y_i = β₀ + ε_i, whose least-squares fit is just ȳ. So the global F-test asks one specific question: does this model, with all its predictors, actually beat just predicting the sample mean for everyone?" },
  { section: 1, type: 'tf',
    q: "A large, significant global F-test statistic tells you which specific predictor(s) are responsible.",
    options: ["True", "False"], correct: 1,
    explain: "The alternative hypothesis is only \"at least one β_j ≠ 0\" — it can't distinguish one strong predictor among ten from all ten being weakly significant. Figuring out <em>which</em> predictor(s) matter is what the partial F-test (and single-coefficient t-tests) are for." },
  { section: 1, type: 'mcq',
    q: "In the general reduced-vs-full formula F = [(SSE_R − SSE_F)/(p−k)] / [SSE_F/(n−p)], what does p − k represent?",
    options: [
      "The number of coefficients (predictors) dropped going from the full model to the reduced model.",
      "The number of observations removed as outliers.",
      "The total number of predictors in the reduced model.",
      "The sample size minus the number of significant predictors."
    ], correct: 0,
    explain: "p is the number of β's in the full model, k is the number of β's in the reduced model, so p − k is exactly how many coefficients got deleted — the numerator degrees of freedom. The global F-test is the special case k = 1 (reduced model keeps only the intercept)." },
  { section: 1, type: 'tf',
    q: "Even though each individual coefficient's null hypothesis (H₀: β_j = 0) is two-sided, the F-test's rejection region sits entirely in the upper tail.",
    options: ["True", "False"], correct: 0,
    explain: "F is a ratio of two non-negative mean squares, so it can never be negative — a large ratio is the only way to get evidence against H₀. There's no \"too small\" side to reject on, so the whole rejection region piles up on the right." },

  // ---- Week 5, Day 1 ----
  { section: 2, type: 'tf',
    q: "For a partial F-test that drops exactly one predictor, the F statistic equals the square of the t-statistic for that coefficient (t² = F).",
    options: ["True", "False"], correct: 0,
    explain: "With one coefficient dropped, the numerator degrees of freedom collapse to 1 — and the two-sided t-test for that single β and the one-predictor partial F-test turn out to be exactly the same test, just written two different ways." },
  { section: 2, type: 'tf',
    q: "Ordinary R² can never decrease when you add another predictor to the model, even a completely useless one.",
    options: ["True", "False"], correct: 0,
    explain: "Adding any column to X can only reduce (or leave unchanged) the achievable SSE, since least squares could always set that new coefficient to exactly 0 if it doesn't help. Since R² = 1 − SSE/SST and SST doesn't change, R² can only go up or stay flat — exactly why R² alone can't compare models of different sizes." },
  { section: 2, type: 'mcq',
    q: "What is the key structural difference between R² and adjusted R²?",
    options: [
      "Adjusted R² divides both SSE and SST by their degrees of freedom before taking the ratio, charging a penalty for every added parameter.",
      "Adjusted R² uses a completely different sum of squares that has nothing to do with SSE.",
      "Adjusted R² is always exactly R² minus 0.05.",
      "Adjusted R² ignores SST entirely and only depends on SSE."
    ], correct: 0,
    explain: "R²_adj = 1 − [SSE/(n−p)] / [SST/(n−1)]. As p grows, n−p shrinks, so SSE/(n−p) tends to grow — while SST/(n−1) doesn't depend on p at all. That imbalance is the entire mechanism of the penalty." },
  { section: 2, type: 'tf',
    q: "Adjusted R² always increases when a new predictor is added, since more information can only help.",
    options: ["True", "False"], correct: 1,
    explain: "It decreases unless the new predictor cuts SSE by enough to outweigh the one degree of freedom it costs. A predictor that barely helps (or doesn't help at all) makes R²_adj go down even as ordinary R² goes up." },
  { section: 2, type: 'mcq',
    q: "In the model log(y_i) = β₀ + β₁X_i + ε_i, how should β₁ be interpreted?",
    options: [
      "Approximately the % change in y per 1-unit change in X.",
      "The elasticity of y with respect to X.",
      "The raw change in y per 1-unit change in X, in y's original units.",
      "The % change in X per 1-unit change in y."
    ], correct: 0,
    explain: "That's specifically the log-y (not log-log) case: logging only the response turns an additive change on the log scale into a multiplicative, percentage-style change back on the original scale. \"Elasticity\" is the log-log case, where both y and X are logged." },
  { section: 2, type: 'tf',
    q: "Once X² is added to a model alongside X, β₁ can still be read as \"the change in y per 1-unit change in X,\" exactly as in a model without X².",
    options: ["True", "False"], correct: 1,
    explain: "You can't move X by one unit while holding X² fixed — they're mechanically linked. With X² in the model, the marginal effect of X becomes β₁ + 2β₂X, a quantity that depends on where you evaluate it, not a single fixed number." },
  { section: 2, type: 'mcq',
    q: "A categorical predictor with K levels is represented in the design matrix using how many dummy (indicator) columns?",
    options: [
      "K − 1",
      "K",
      "K + 1, to also include a column for the reference level",
      "Always exactly 2, regardless of K"
    ], correct: 0,
    explain: "Giving all K levels their own column would make those K columns sum to exactly the intercept column (every row has a 1 in exactly one of them), which is the same full-rank problem from question 2 — X<sup>T</sup>X would be singular. One level is absorbed into the intercept with no column of its own — the reference level — and every other coefficient is read as a difference from it." },
];

const SECTIONS = [
  "Week 4, Day 1 — Matrix Notation & Multiple Regression",
  "Week 4, Day 2 — The Hat Matrix & the F-test Family",
  "Week 5, Day 1 — Partial F-tests, Adjusted R², & Feature Engineering",
];

const STORAGE_KEY = "mlr-fundamentals-quiz-v2"; // v2: options are shuffled, answers stored by text not index
let answered = {};
try {
  answered = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
} catch (e) { answered = {}; }

function saveProgress() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(answered)); } catch (e) { /* ignore */ }
}

const letters = ["A", "B", "C", "D"];

// Shuffle each MCQ's options at load time so the correct answer isn't
// predictably in the same position (question-writing tends to default to
// "correct answer first" unless actively randomized). Answers are recorded
// by their text, not their index, so this is safe to do on every page load
// without corrupting anything saved from a previous session.
function shuffle(array) {
  const arr = array.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
QUESTIONS.forEach(item => {
  if (item.type === "mcq") {
    const correctText = item.options[item.correct];
    item.options = shuffle(item.options);
    item.correct = item.options.indexOf(correctText);
  }
});

function render() {
  const quiz = document.getElementById("quiz");
  quiz.innerHTML = "";
  let currentSection = -1;

  QUESTIONS.forEach((item, idx) => {
    if (item.section !== currentSection) {
      currentSection = item.section;
      const h = document.createElement("div");
      h.className = "section-header";
      h.innerHTML = `<div class="section-eyebrow">Section ${currentSection + 1} of ${SECTIONS.length}</div><h2>${SECTIONS[currentSection]}</h2>`;
      quiz.appendChild(h);
    }

    const card = document.createElement("div");
    card.className = "card";

    const meta = document.createElement("div");
    meta.className = "q-meta";
    meta.innerHTML = `<span class="q-num">Q${idx + 1}</span><span class="q-type">${item.type === "tf" ? "True / False" : "Multiple choice"}</span>`;
    card.appendChild(meta);

    const qText = document.createElement("p");
    qText.className = "q-text";
    qText.innerHTML = item.q;
    card.appendChild(qText);

    const optWrap = document.createElement("div");
    optWrap.className = "options";

    const chosen = answered[idx]; // { text, correct } or undefined

    item.options.forEach((optText, optIdx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "opt";
      btn.innerHTML = `<span class="opt-letter">${letters[optIdx]}</span><span>${optText}</span>`;
      if (chosen !== undefined) {
        btn.disabled = true;
        if (optIdx === item.correct) btn.classList.add("is-correct");
        else if (optText === chosen.text) btn.classList.add("is-wrong");
      }
      btn.addEventListener("click", () => {
        answered[idx] = { text: optText, correct: optIdx === item.correct };
        saveProgress();
        render();
        updateScore();
      });
      optWrap.appendChild(btn);
    });
    card.appendChild(optWrap);

    const fb = document.createElement("div");
    const isCorrect = chosen !== undefined && chosen.correct;
    fb.className = "feedback" + (chosen !== undefined ? " show " + (isCorrect ? "correct" : "incorrect") : "");
    fb.innerHTML = `<span class="verdict">${isCorrect ? "Correct" : "Not quite"}</span>${item.explain}`;
    card.appendChild(fb);

    quiz.appendChild(card);
  });
}

function updateScore() {
  const total = QUESTIONS.length;
  const answeredCount = Object.keys(answered).length;
  const correctCount = Object.values(answered).filter(a => a.correct).length;
  const pill = document.getElementById("scorePill");
  pill.textContent = answeredCount === total
    ? `${correctCount} / ${total} correct`
    : `${answeredCount} / ${total} answered`;
}

document.getElementById("resetBtn").addEventListener("click", () => {
  answered = {};
  saveProgress();
  render();
  updateScore();
});

render();
updateScore();
