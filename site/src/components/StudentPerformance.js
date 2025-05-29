// site/src/components/StudentPerformance.js
import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { INSTRUMENTATION, NUMERIC_COLS } from "../constants";
import "./StudentPerformance.css";

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
  const parentCounts = categoricalCounts?.Parental_Involvement ?? null;

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

  return (
    <div className="student-perf">
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
              <th>Std. Dev.</th>
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

      {/* Charts */}
      <div className="chart-container">
        <h3>Exam Score Distribution</h3>
        {examHist ? (
          <Bar
            data={{
              labels: examHist.labels,
              datasets: [
                {
                  label: "Count",
                  data: examHist.counts,
                  backgroundColor: "rgba(75, 192, 192, 0.6)",
                  borderColor: "rgba(75, 192, 192, 1)",
                  borderWidth: 1,
                },
              ],
            }}
            options={{
              plugins: {
                legend: { display: false },
                title: { display: true, text: "Histogram of Exam Scores" },
              },
              scales: {
                x: { title: { display: true, text: "Score Range" } },
                y: { title: { display: true, text: "Frequency" } },
              },
            }}
          />
        ) : (
          <p className="loading">Preparing histogram…</p>
        )}
      </div>

      <div className="chart-container">
        <h3>Parental Involvement Levels</h3>
        {parentCounts ? (
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
        <h3>Hours Studied vs. Exam Score</h3>
        {scatterData ? (
          <Scatter
            data={scatterData}
            options={{
              plugins: {
                title: { display: true, text: "Scatter: Hours vs Exam Score" },
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
        <h3>Average Exam Score by School Type</h3>
        {schoolData ? (
          <Bar
            data={schoolData}
            options={{
              plugins: {
                title: { display: true, text: "Avg Score by School Type" },
              },
              scales: {
                x: { title: { display: true, text: "School Type" } },
                y: { title: { display: true, text: "Avg Exam Score" } },
              },
            }}
          />
        ) : (
          <p className="loading">Preparing average‐by‐school chart…</p>
        )}
      </div>
    </div>
  );
}
