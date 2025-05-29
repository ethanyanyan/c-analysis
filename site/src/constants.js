export const INSTRUMENTATION = [
  {
    name: "Hours_Studied",
    desc: "Hours spent studying per week",
    scale: "Ratio",
  },
  { name: "Attendance", desc: "Percent of classes attended", scale: "Ratio" },
  {
    name: "Parental_Involvement",
    desc: "Parental involvement level",
    scale: "Ordinal (Low→High)",
  },
  {
    name: "Access_to_Resources",
    desc: "Availability of educational resources",
    scale: "Ordinal (Low→High)",
  },
  {
    name: "Extracurricular_Activities",
    desc: "Participates in activities",
    scale: "Nominal (Yes/No)",
  },
  { name: "Sleep_Hours", desc: "Hours of sleep per night", scale: "Ratio" },
  {
    name: "Previous_Scores",
    desc: "Scores from previous exams",
    scale: "Ratio",
  },
  {
    name: "Motivation_Level",
    desc: "Student motivation",
    scale: "Ordinal (Low→High)",
  },
  {
    name: "Internet_Access",
    desc: "Has internet access",
    scale: "Nominal (Yes/No)",
  },
  {
    name: "Tutoring_Sessions",
    desc: "Tutoring sessions per month",
    scale: "Ratio",
  },
  {
    name: "Family_Income",
    desc: "Family income level",
    scale: "Ordinal (Low→High)",
  },
  {
    name: "Teacher_Quality",
    desc: "Quality of teachers",
    scale: "Ordinal (Low→High)",
  },
  { name: "School_Type", desc: "Type of school", scale: "Nominal" },
  {
    name: "Peer_Influence",
    desc: "Peer influence on performance",
    scale: "Nominal",
  },
  {
    name: "Physical_Activity",
    desc: "Hours of physical activity per week",
    scale: "Ratio",
  },
  {
    name: "Learning_Disabilities",
    desc: "Has learning disability",
    scale: "Nominal",
  },
  {
    name: "Parental_Education_Level",
    desc: "Highest parental education",
    scale: "Ordinal",
  },
  {
    name: "Distance_from_Home",
    desc: "Distance home→school",
    scale: "Ordinal",
  },
  { name: "Gender", desc: "Student gender", scale: "Nominal" },
  { name: "Exam_Score", desc: "Final exam score", scale: "Ratio" },
];

export const NUMERIC_COLS = [
  "Hours_Studied",
  "Attendance",
  "Sleep_Hours",
  "Previous_Scores",
  "Tutoring_Sessions",
  "Physical_Activity",
  "Exam_Score",
];
