// site/src/components/BikeSharing.js

import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./BikeSharing.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function BikeSharing() {
  const [rows, setRows] = useState(null);
  const [numericStats, setNumericStats] = useState(null);
  const [categoricalCounts, setCategoricalCounts] = useState(null);
  const [timeSeries, setTimeSeries] = useState(null);

  useEffect(() => {
    Papa.parse(`${process.env.PUBLIC_URL}/data/train.csv`, {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: ({ data }) => {
        // filter out any bad rows
        const clean = data.filter((r) => r.count != null && r.datetime);
        setRows(clean);
        setNumericStats(
          computeNumericStats(clean, [
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "count",
          ])
        );
        setCategoricalCounts(
          computeCategoricalCounts(clean, [
            "season",
            "holiday",
            "workingday",
            "weather",
          ])
        );
        setTimeSeries(buildTimeSeries(clean));
      },
      error: (err) => console.error("Failed to parse CSV:", err),
    });
  }, []);

  function computeNumericStats(data, cols) {
    const stats = {};
    cols.forEach((col) => {
      const vals = data
        .map((r) => r[col])
        .filter((v) => typeof v === "number" && !isNaN(v))
        .sort((a, b) => a - b);
      if (!vals.length) return;
      const n = vals.length;
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / n;
      const median = (vals[((n - 1) / 2) | 0] + vals[(n / 2) | 0]) / 2;
      const min = vals[0];
      const max = vals[n - 1];
      const sd = Math.sqrt(
        vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
      );
      const q1 = vals[Math.floor((n + 1) / 4) - 1] ?? min;
      const q3 = vals[Math.ceil((3 * (n + 1)) / 4) - 1] ?? max;
      stats[col] = { n, mean, median, min, q1, q3, max, sd };
    });
    return stats;
  }

  function computeCategoricalCounts(data, cols) {
    const counts = {};
    cols.forEach((col) => (counts[col] = {}));
    data.forEach((r) => {
      cols.forEach((col) => {
        const v = r[col] != null ? String(r[col]) : "Missing";
        counts[col][v] = (counts[col][v] || 0) + 1;
      });
    });
    return counts;
  }

  function buildHistogram(data, col, bins = 15) {
    const values = data.map((r) => r[col]).filter((v) => typeof v === "number");
    if (!values.length) return null;
    const min = Math.min(...values),
      max = Math.max(...values),
      step = (max - min) / bins;
    const counts = Array(bins).fill(0);
    values.forEach((v) => {
      const i = Math.min(bins - 1, Math.floor((v - min) / step));
      counts[i]++;
    });
    const labels = counts.map(
      (_, i) =>
        `${(min + i * step).toFixed(1)}–${(min + (i + 1) * step).toFixed(1)}`
    );
    return { labels, counts };
  }

  function buildTimeSeries(data) {
    const sorted = [...data].sort(
      (a, b) => new Date(a.datetime) - new Date(b.datetime)
    );
    return {
      labels: sorted.map((r) => r.datetime),
      counts: sorted.map((r) => r.count),
      seasons: sorted.map((r) => r.season),
    };
  }

  // Prepare chart data
  const countHist = rows ? buildHistogram(rows, "count", 15) : null;
  const ts = timeSeries;
  const avgBySeason = rows
    ? (() => {
        const agg = rows.reduce((acc, r) => {
          const s = r.season;
          if (!acc[s]) acc[s] = { sum: 0, count: 0 };
          acc[s].sum += r.count;
          acc[s].count++;
          return acc;
        }, {});
        const seasonMap = {
          1: "Spring",
          2: "Summer",
          3: "Fall",
          4: "Winter",
        };
        const labels = Object.keys(agg).map((s) => seasonMap[s] || s);
        const data = Object.keys(agg).map((s) => agg[s].sum / agg[s].count);
        return { labels, data };
      })()
    : null;

  const seasonMap = { 1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter" };
  const seasonColors = {
    Spring: "#4daf4a",
    Summer: "#377eb8",
    Fall: "#ff7f00",
    Winter: "#984ea3",
  };

  const labels = timeSeries?.labels ?? [];
  const counts = timeSeries?.counts ?? [];
  const seasons = timeSeries?.seasons ?? [];

  // if rows might be null at first, guard it
  const uniqueSeasons = timeSeries ? [...new Set(seasons)] : [];

  const seasonalDatasets = uniqueSeasons.map((seasonKey) => {
    const seasonName = seasonMap[seasonKey];
    return {
      label: seasonName,
      data: labels.map((_, idx) =>
        seasons[idx] === seasonKey ? counts[idx] : null
      ),
      borderColor: seasonColors[seasonName],
      spanGaps: true,
      fill: false,
    };
  });

  return (
    <div className="bike-sharing">
      <h2>Bike Sharing Demand</h2>

      <section id="dataset-info">
        <h3>Data Set Information</h3>
        <p>
          Bike sharing systems are a means of renting bicycles where the process
          of obtaining membership, rental, and bike return is automated via a
          network of kiosk locations throughout a city. … Participants forecast
          bike rental demand in the Capital Bikeshare program in Washington,
          D.C.
        </p>
        <p>
          <strong>Acknowledgements:</strong> Provided by Hadi Fanaee Tork (UCI
          ML Repo). <strong>Citation:</strong> Fanaee-T &amp; Gama (2013),
          Progress in Artificial Intelligence.
        </p>
      </section>

      <section id="numeric">
        <h3>Numeric Descriptives</h3>
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
                  <td>{s.min.toFixed(2)}</td>
                  <td>{s.q1.toFixed(2)}</td>
                  <td>{s.q3.toFixed(2)}</td>
                  <td>{s.max.toFixed(2)}</td>
                  <td>{s.sd.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>Loading numeric summaries…</p>
        )}
      </section>

      <section id="categorical">
        <h3>Categorical Frequencies</h3>
        {categoricalCounts ? (
          Object.entries(categoricalCounts).map(([col, cnts]) => (
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
                  {Object.entries(cnts).map(([val, c]) => (
                    <tr key={val}>
                      <td>{val}</td>
                      <td>{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))
        ) : (
          <p>Loading categorical frequencies…</p>
        )}
      </section>

      <section id="charts">
        <h3>Charts</h3>

        <div className="chart-container">
          <h4>Histogram of Total Rentals</h4>
          {countHist ? (
            <Bar
              data={{
                labels: countHist.labels,
                datasets: [{ label: "Count", data: countHist.counts }],
              }}
              options={{
                plugins: {
                  title: { display: true, text: "Histogram of Total Rentals" },
                },
                scales: {
                  x: { title: { display: true, text: "Rental Count" } },
                  y: { title: { display: true, text: "Frequency" } },
                },
              }}
            />
          ) : (
            <p>Preparing histogram…</p>
          )}
        </div>

        <div className="chart-container">
          <h4>Hourly Rentals Over Time</h4>
          {timeSeries ? (
            <Line
              data={{
                labels: labels,
                datasets: [
                  {
                    label: "Count",
                    data: counts,
                    borderWidth: 2,
                    borderColor: (ctx) => {
                      const idx = ctx.p0DataIndex;
                      const seasonKey = seasons[idx];
                      return seasonColors[seasonMap[seasonKey]] || "#888";
                    },
                    segment: { borderWidth: 2 },
                    fill: false,
                  },
                ],
              }}
              options={{
                plugins: {
                  title: { display: true, text: "Rentals Over Time" },
                  legend: { display: false },
                },
                scales: {
                  x: { title: { display: true, text: "Datetime" } },
                  y: { title: { display: true, text: "Count" } },
                },
              }}
            />
          ) : (
            <p>Preparing time series…</p>
          )}
        </div>

        <div className="chart-container">
          <h4>Average Rentals by Season</h4>
          {avgBySeason ? (
            <Bar
              data={{
                labels: avgBySeason.labels,
                datasets: [{ label: "Avg Count", data: avgBySeason.data }],
              }}
              options={{
                plugins: {
                  title: {
                    display: true,
                    text: "Average Bike Rentals by Season",
                  },
                },
                scales: {
                  x: { title: { display: true, text: "Season" } },
                  y: { title: { display: true, text: "Avg Count" } },
                },
              }}
            />
          ) : (
            <p>Preparing seasonal bar chart…</p>
          )}
        </div>
      </section>
    </div>
  );
}
