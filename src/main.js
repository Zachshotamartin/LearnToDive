import { mountExperiment } from "./index.js";
import "./style.css";
const app = mountExperiment(document.querySelector("#app"));
if (import.meta.hot) import.meta.hot.dispose(() => app.dispose());
