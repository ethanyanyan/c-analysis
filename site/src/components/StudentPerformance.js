// site/src/components/StudentPerformance.js
import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { INSTRUMENTATION, NUMERIC_COLS } from "../constants";
import "./StudentPerformance.css";
import PredictModal from "./PredictModal";

// Chart imports
import { Bar, Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

export default function StudentPerformance() {
  const [rows, setRows] = useState(null);
  const [numericStats, setNumericStats] = useState(null);
  const [categoricalCounts, setCategoricalCounts] = useState(null);
  const [xKey, setXKey] = useState("Hours_Studied");
  const [yKey, setYKey] = useState("Exam_Score");
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    Papa.parse(`${process.env.PUBLIC_URL}/data/StudentPerformanceFactors.csv`, {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: ({ data }) => {
        const clean = data.filter((r) => r.Exam_Score != null);
        console.log(clean);
        setRows(clean);
        setNumericStats(computeNumericStats(clean));
        setCategoricalCounts(computeCategoricalCounts(clean));
      },
      error: (err, file, inputElem, reason) => {
        console.error("Failed to parse CSV:", reason);
      },
    });
  }, []);

  // Numeric summaries
  function computeNumericStats(data) {
    const stats = {};
    NUMERIC_COLS.forEach((col) => {
      const vals = data
        .map((r) => r[col])
        .filter((v) => typeof v === "number" && !isNaN(v))
        .sort((a, b) => a - b);

      if (!vals.length) return;
      const n = vals.length;
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / n;
      const median = (vals[((n - 1) / 2) | 0] + vals[(n / 2) | 0]) / 2;
      const q1 = vals[Math.floor((n + 1) / 4) - 1] ?? vals[0];
      const q3 = vals[Math.ceil((3 * (n + 1)) / 4) - 1] ?? vals[n - 1];
      const sd = Math.sqrt(
        vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
      );

      stats[col] = {
        n,
        mean,
        median,
        min: vals[0],
        q1,
        q3,
        max: vals[n - 1],
        sd,
      };
    });
    return stats;
  }

  // Categorical counts
  function computeCategoricalCounts(data) {
    const cats = {};
    INSTRUMENTATION.filter((c) => !NUMERIC_COLS.includes(c.name)).forEach(
      (c) => (cats[c.name] = {})
    );
    data.forEach((r) => {
      Object.keys(cats).forEach((col) => {
        const v = r[col] ?? "Missing";
        cats[col][v] = (cats[col][v] || 0) + 1;
      });
    });
    return cats;
  }

  // Build simple histogram data
  function buildHistogram(rows, col, bins = 10) {
    const values = rows.map((r) => r[col]).filter((v) => typeof v === "number");
    if (!values.length) return null;
    const min = Math.min(...values),
      max = Math.max(...values);
    const step = (max - min) / bins;
    const counts = Array(bins).fill(0);
    values.forEach((v) => {
      const idx = Math.min(bins - 1, Math.floor((v - min) / step));
      counts[idx]++;
    });
    const labels = counts.map(
      (_, i) =>
        `${(min + i * step).toFixed(1)}–${(min + (i + 1) * step).toFixed(1)}`
    );
    return { labels, counts };
  }

  const examHist = rows ? buildHistogram(rows, "Exam_Score", 15) : null;
  const parentCounts = categoricalCounts?.Parental_Involvement || {};

  // scatter: Hours_Studied vs Exam_Score
  const scatterData = rows
    ? {
        datasets: [
          {
            label: "Hours vs Exam",
            data: rows.map((r) => ({
              x: r.Hours_Studied,
              y: r.Exam_Score,
            })),
            backgroundColor: "rgba(255, 99, 132, 0.6)",
          },
        ],
      }
    : null;

  // bar: avg Exam_Score by School_Type
  const schoolData = rows
    ? (() => {
        const agg = rows.reduce((acc, r) => {
          if (!acc[r.School_Type]) acc[r.School_Type] = { sum: 0, count: 0 };
          acc[r.School_Type].sum += r.Exam_Score;
          acc[r.School_Type].count++;
          return acc;
        }, {});
        const labels = Object.keys(agg);
        const avgs = labels.map((k) => agg[k].sum / agg[k].count);
        return {
          labels,
          datasets: [
            {
              label: "Avg Exam Score",
              data: avgs,
              backgroundColor: [
                "rgba(54, 162, 235, 0.6)",
                "rgba(75, 192, 192, 0.6)",
              ],
            },
          ],
        };
      })()
    : null;

  // Linear regression helper
  function computeRegression(data, xKey, yKey) {
    const filtered = data.filter(
      (r) =>
        typeof r[xKey] === "number" &&
        !isNaN(r[xKey]) &&
        typeof r[yKey] === "number" &&
        !isNaN(r[yKey])
    );
    const n = filtered.length;
    const xbar = filtered.reduce((sum, r) => sum + r[xKey], 0) / n;
    const ybar = filtered.reduce((sum, r) => sum + r[yKey], 0) / n;
    let ssxx = 0,
      ssxy = 0,
      ssyy = 0;
    filtered.forEach((r) => {
      const dx = r[xKey] - xbar,
        dy = r[yKey] - ybar;
      ssxx += dx * dx;
      ssxy += dx * dy;
      ssyy += dy * dy;
    });
    const slope = ssxy / ssxx;
    const intercept = ybar - slope * xbar;
    const ssr = slope * ssxy;
    const r2 = ssr / ssyy;
    return { slope, intercept, r2, n };
  }

  const reg = rows
    ? computeRegression(rows, "Hours_Studied", "Exam_Score")
    : null;

  // 4) Hard‐coded two‐sample t‐test results
  const tTest = {
    public: { M: 67.21, SD: 3.91, n: 4598 },
    private: { M: 67.29, SD: 3.85, n: 2009 },
    t: -0.72,
    df: 6605,
    p: 0.472,
    d: 0.02,
  };

  // 5) Hard‐coded multivariable regression results
  const multiReg = {
    intercept: 42.426,
    hours: 0.294,
    attendance: 0.198,
    motivation: 0.523,
    resources: 0.994,
    r2: 0.582,
    n: 6607,
  };

  // 6) Hard‐coded one‐way ANOVA results
  const anovaTest = {
    F: 84.49,
    dfBetween: 2,
    dfWithin: 6604,
    p: 5.875479e-37,
    eta2: 0.025,
    groups: {
      High: { mean: 68.09, sd: 3.95, n: 1908 },
      Medium: { mean: 67.1, sd: 3.73, n: 3362 },
      Low: { mean: 66.36, sd: 3.97, n: 1337 },
    },
  };

  // Prediction

  return (
    <div className="student-perf">
      <button onClick={() => setModalOpen(true)}>
        Predict with ElasticNet Model
      </button>

      <PredictModal
        isOpen={modalOpen}
        onRequestClose={() => setModalOpen(false)}
      />

      <h2>Student Performance Factors</h2>
      <p>
        Dataset URL:&nbsp;
        <a
          href="https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"
          target="_blank"
          rel="noopener noreferrer"
        >
          Kaggle: Student Performance Factors
        </a>
      </p>

      {/* ─── Table of Contents ───────────────────────────────────── */}
      <nav className="toc">
        <h2>Table of Contents</h2>
        <ul>
          <li>
            <a href="#instrumentation">1. Instrumentation</a>
          </li>
          <li>
            <a href="#numeric">2. Numeric Descriptives</a>
          </li>
          <li>
            <a href="#categorical">3. Categorical Frequencies</a>
          </li>
          <li>
            <a href="#charts">4. Charts</a>
          </li>
          <li>
            <a href="#inferential">5. Inferential & Modeling</a>
          </li>
          <li>
            <a href="#interpretation">6. Interpretation & Reporting</a>
          </li>
        </ul>
      </nav>

      {/* ─── 1. Instrumentation ───────────────────────────────────── */}
      <section id="instrumentation">
        <h3>1. Instrumentation</h3>
        <table>
          <thead>
            <tr>
              <th>Attribute</th>
              <th>Description</th>
              <th>Scale</th>
            </tr>
          </thead>
          <tbody>
            {INSTRUMENTATION.map((c) => (
              <tr key={c.name}>
                <td>{c.name}</td>
                <td>{c.desc}</td>
                <td>{c.scale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* ─── 2. Numeric Descriptives ─────────────────────────────── */}
      <section id="numeric">
        <h3>2. Numeric Descriptives</h3>
        {numericStats ? (
          <table>
            <thead>
              <tr>
                <th>Variable</th>
                <th>n</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Min</th>
                <th>Q1</th>
                <th>Q3</th>
                <th>Max</th>
                <th>SD</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(numericStats).map(([col, s]) => (
                <tr key={col}>
                  <td>{col}</td>
                  <td>{s.n}</td>
                  <td>{s.mean.toFixed(2)}</td>
                  <td>{s.median.toFixed(2)}</td>
                  <td>{s.min}</td>
                  <td>{s.q1.toFixed(2)}</td>
                  <td>{s.q3.toFixed(2)}</td>
                  <td>{s.max}</td>
                  <td>{s.sd.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="loading">Loading numeric summaries…</p>
        )}
      </section>

      {/* ─── 3. Categorical Frequencies ───────────────────────────── */}
      <section id="categorical">
        <h3>3. Categorical Frequencies</h3>
        {categoricalCounts ? (
          Object.entries(categoricalCounts).map(([col, counts]) => (
            <div key={col}>
              <h4>{col}</h4>
              <table>
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(counts).map(([val, cnt]) => (
                    <tr key={val}>
                      <td>{val}</td>
                      <td>{cnt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))
        ) : (
          <p className="loading">Loading categorical frequencies…</p>
        )}
      </section>

      {/* ─── 4. Charts ────────────────────────────────────────────── */}
      <section id="charts">
        <h3>4. Charts</h3>
        <div className="chart-container">
          {/* ─── Variable selectors for regression ─────────────────── */}
          <div style={{ margin: "1rem 0" }}>
            <label>
              X variable:&nbsp;
              <select value={xKey} onChange={(e) => setXKey(e.target.value)}>
                {NUMERIC_COLS.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
            &nbsp;&nbsp;
            <label>
              Y variable:&nbsp;
              <select value={yKey} onChange={(e) => setYKey(e.target.value)}>
                {NUMERIC_COLS.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <h4>
            {yKey} vs. {xKey}
          </h4>
          {rows ? (
            (() => {
              // recompute regression & scatter on-the-fly
              const reg = computeRegression(rows, xKey, yKey);
              const scatter = rows.map((r) => ({ x: r[xKey], y: r[yKey] }));
              // line endpoints at min/max x
              const xs = scatter.map((p) => p.x);
              const minX = Math.min(...xs),
                maxX = Math.max(...xs);
              const line = [
                { x: minX, y: reg.intercept + reg.slope * minX },
                { x: maxX, y: reg.intercept + reg.slope * maxX },
              ];
              return (
                <Scatter
                  data={{
                    datasets: [
                      {
                        label: `${yKey} vs ${xKey}`,
                        data: scatter,
                        // light blue points
                        backgroundColor: "rgba(173, 216, 230, 0.6)",
                        borderColor: "rgba(173, 216, 230, 1)",
                        pointBackgroundColor: "rgba(173, 216, 230, 0.6)",
                        pointBorderColor: "rgba(173, 216, 230, 1)",
                      },
                      {
                        label: "Fit line",
                        data: line,
                        showLine: true,
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 2,
                        // distinct red line
                        borderColor: "rgba(255, 0, 0, 1)",
                        backgroundColor: "rgba(255, 0, 0, 0.1)",
                      },
                    ],
                  }}
                  options={{
                    plugins: {
                      title: { display: true, text: `${yKey} vs. ${xKey}` },
                    },
                    scales: {
                      x: { title: { display: true, text: xKey } },
                      y: { title: { display: true, text: yKey } },
                    },
                  }}
                />
              );
            })()
          ) : (
            <p className="loading">Preparing scatter…</p>
          )}
        </div>

        <div className="chart-container">
          <h4>Parental Involvement Levels</h4>
          {Object.keys(parentCounts).length > 0 ? (
            <Bar
              data={{
                labels: Object.keys(parentCounts),
                datasets: [
                  {
                    label: "Students",
                    data: Object.values(parentCounts),
                    backgroundColor: [
                      "rgba(255, 159, 64, 0.6)",
                      "rgba(153, 102, 255, 0.6)",
                      "rgba(255, 205, 86, 0.6)",
                    ],
                    borderColor: [
                      "rgba(255, 159, 64, 1)",
                      "rgba(153, 102, 255, 1)",
                      "rgba(255, 205, 86, 1)",
                    ],
                    borderWidth: 1,
                  },
                ],
              }}
              options={{
                plugins: {
                  title: { display: true, text: "Parental Involvement" },
                },
                scales: {
                  x: { title: { display: true, text: "Level" } },
                  y: { title: { display: true, text: "Count" } },
                },
              }}
            />
          ) : (
            <p className="loading">Preparing bar chart…</p>
          )}
        </div>

        <div className="chart-container">
          <h4>Hours Studied vs. Exam Score</h4>
          {scatterData ? (
            <Scatter
              data={scatterData}
              options={{
                plugins: {
                  title: {
                    display: true,
                    text: "Scatter: Hours Studied vs. Exam Score",
                  },
                },
                scales: {
                  x: {
                    type: "linear",
                    position: "bottom",
                    title: { display: true, text: "Hours Studied" },
                  },
                  y: { title: { display: true, text: "Exam Score" } },
                },
              }}
            />
          ) : (
            <p className="loading">Preparing scatter…</p>
          )}
        </div>

        <div className="chart-container">
          <h4>Avg. Exam Score by School Type</h4>
          {schoolData ? (
            <Bar
              data={schoolData}
              options={{
                plugins: {
                  title: {
                    display: true,
                    text: "Average Exam Score by School Type",
                  },
                },
                scales: {
                  x: { title: { display: true, text: "School Type" } },
                  y: {
                    title: { display: true, text: "Avg Exam Score" },
                  },
                },
              }}
            />
          ) : (
            <p className="loading">Preparing school‐type chart…</p>
          )}
        </div>
      </section>

      {/* ─────────── 5. Inferential & Modeling ─────────── */}
      <section id="inferential">
        <h3>5. Inferential & Modeling Techniques</h3>

        {/* 5.1 Two-Sample t-Test */}
        <article>
          <h4>5.1 Two-Sample t-Test: Public vs. Private School</h4>
          <h5>Results</h5>
          <p>
            Public: M = {tTest.public.M.toFixed(2)}, SD ={" "}
            {tTest.public.SD.toFixed(2)}, n = {tTest.public.n}
            <br />
            Private: M = {tTest.private.M.toFixed(2)}, SD ={" "}
            {tTest.private.SD.toFixed(2)}, n = {tTest.private.n}
            <br />
            t({tTest.df}) = {tTest.t.toFixed(2)}, p = {tTest.p.toFixed(3)},
            Cohen’s d = {tTest.d.toFixed(2)}
          </p>
          <h5>Interpretation</h5>
          <p>
            There was <strong>no significant difference</strong> in exam scores
            between public and private school students, t({tTest.df}) ={" "}
            {tTest.t.toFixed(2)}, p = {tTest.p.toFixed(3)}. The effect size was
            negligible (d = {tTest.d.toFixed(2)}).
          </p>
        </article>

        {/* 5.2 Simple Regression */}
        <article>
          <h4>5.2 Simple Linear Regression</h4>
          {reg ? (
            <>
              <img
                src={`${process.env.PUBLIC_URL}/images/simple_regression.png`}
                alt="Exam vs Hours"
              />
              <h5>Results</h5>
              <p>
                β₁ = {reg.slope.toFixed(2)}
                <br />
                Intercept = {reg.intercept.toFixed(2)}
                <br />
                R² = {(reg.r2 * 100).toFixed(1)}%<br />n = {reg.n}
              </p>
              <h5>Interpretation</h5>
              <p>
                Each additional hour of study per week is associated with a{" "}
                {reg.slope.toFixed(2)}-point increase in exam score on average.
                However, the model explains only {(reg.r2 * 100).toFixed(1)}% of
                the variance, indicating that other factors also play a
                substantial role.
              </p>
            </>
          ) : (
            <p className="loading">Computing regression…</p>
          )}
        </article>

        {/* 5.3 Multi-variate regression */}
        <article id="multiregression">
          <h4>
            5.3 Multiple Regression: Exam Score ~ Hours + Attendance +
            Motivation + Resources
          </h4>
          <img
            src={`${process.env.PUBLIC_URL}/images/multi_regression_residuals.png`}
            alt="Residuals vs Fitted"
          />
          <h5>Results</h5>
          <p>
            Intercept = {multiReg.intercept.toFixed(3)}
            <br />
            Hours Studied β = {multiReg.hours.toFixed(3)}
            <br />
            Attendance β = {multiReg.attendance.toFixed(3)}
            <br />
            Motivation_Num β = {multiReg.motivation.toFixed(3)}
            <br />
            Resources_Num β = {multiReg.resources.toFixed(3)}
            <br />
            R² = {(multiReg.r2 * 100).toFixed(1)}%, n = {multiReg.n}
          </p>
          <h5>Interpretation</h5>
          <p>
            All four predictors were significant (p &lt; .001). A one-hour
            increase in study time corresponds to an additional ~0.29 points on
            the exam, controlling for attendance, motivation, and resource
            access. Together these explain ~58.2% of score variance.
          </p>
        </article>

        {/* 5.4 One-way ANOVA */}
        <article id="anova">
          <h4>5.4 One-Way ANOVA: Exam Score ~ Parental Involvement</h4>
          <img
            src={`${process.env.PUBLIC_URL}/images/anova_boxplot.png`}
            alt="Exam Score by Parental Involvement"
          />
          <h5>Results</h5>
          <p>
            F({anovaTest.dfBetween}, {anovaTest.dfWithin}) ={" "}
            {anovaTest.F.toFixed(2)}, p &lt; .001, η² ={" "}
            {anovaTest.eta2.toFixed(3)}
          </p>
          <h5>Group Means ± SD (n)</h5>
          <ul>
            {Object.entries(anovaTest.groups).map(([lvl, g]) => (
              <li key={lvl}>
                {lvl}: {g.mean.toFixed(2)} ± {g.sd.toFixed(2)} (n = {g.n})
              </li>
            ))}
          </ul>
          <h5>Interpretation</h5>
          <p>
            There is a small but significant effect of parental involvement on
            exam scores, F({anovaTest.dfBetween}, {anovaTest.dfWithin}) ={" "}
            {anovaTest.F.toFixed(2)}, p &lt; .001, η² ={" "}
            {anovaTest.eta2.toFixed(3)}. Higher involvement corresponds to
            higher average scores.
          </p>
        </article>
      </section>

      {/* ─── 6. Interpretation & Reporting ───────────────────────── */}
      <section id="interpretation">
        <h3>6. Interpretation & Reporting</h3>

        <article>
          <h4>6.1 Two-Sample t-Test</h4>
          <p>
            Public-school students (M = 67.21 ± 3.91, n = 4598) versus
            private-school students (M = 67.29 ± 3.85, n = 2009) did{" "}
            <strong>not</strong> differ significantly on exam score, t(6605) =
            –0.72, p = .472, d = .02.
          </p>
        </article>

        <article>
          <h4>6.2 Simple Linear Regression</h4>
          <p>
            Exam_Score ~ Hours_Studied: β₁ = .29, intercept = 61.46, R² = 19.8%,
            n = 6607. Each extra hour of study is associated with a .29-point
            increase on the exam, but the model explains only ≈ 20% of score
            variance.
          </p>
        </article>

        <article>
          <h4>6.3 Multiple Regression</h4>
          <p>
            Exam_Score ~ Hours_Studied + Attendance + Motivation_Num +
            Resources_Num: intercept = 42.43; βhours = .294; βattendance = .198;
            βmotivation = .523; βresources = .994; R² = 58.2%, n = 6607. All
            predictors are highly significant (p &lt; .001). Together they
            explain a majority of score variance.
          </p>
        </article>

        <article>
          <h4>6.4 One-Way ANOVA</h4>
          <p>
            Exam_Score ~ Parental_Involvement: F(2, 6604) = 84.49, p &lt; .001,
            η² = .025. Group means (± SD): High 68.09 ± 3.95, Medium 67.10 ±
            3.73, Low 66.36 ± 3.97. There is a small but significant effect:
            higher parental involvement corresponds to higher exam performance.
          </p>
        </article>

        <h4>6.5 Overall Discussion</h4>
        <p>
          Across all analyses: no detectable advantage by school type, but clear
          positive associations of study habits, attendance, personal
          motivation, and resource access with exam performance. Parental
          involvement also showed a modest but significant effect.
        </p>

        <h4>6.6 Limitations</h4>
        <ul>
          <li>
            Observational data subject to confounding (income,
            extracurriculars).
          </li>
          <li>
            Assumptions of normality, linearity, and homoscedasticity may be
            violated.
          </li>
          <li>No corrections for multiple comparisons.</li>
        </ul>

        <h4>6.7 Recommendations & Future Work</h4>
        <ul>
          <li>
            Extend to hierarchical/mixed models to account for school
            clustering.
          </li>
          <li>
            Include additional covariates (socioeconomic status, extracurricular
            participation).
          </li>
          <li>
            Explore nonlinear effects and interaction terms (e.g. Hours ×
            Resources).
          </li>
        </ul>
      </section>
    </div>
  );
}
