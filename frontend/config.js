// Optional runtime config for the static frontend.
// Leave PAPERGRAPH_API_BASE empty when the API is served from the same
// origin (Vercel with api/, or the Docker image). Set it to the API's URL
// when the frontend and API are hosted separately, e.g.
//   window.PAPERGRAPH_API_BASE = "https://papergraph-api-xxxx.a.run.app";
window.PAPERGRAPH_API_BASE = window.PAPERGRAPH_API_BASE || "";
