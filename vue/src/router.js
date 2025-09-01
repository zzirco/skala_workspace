import { createRouter, createWebHistory } from "vue-router";
import DisguiseMenu from "../pages/DisguiseMenu.vue";
import DisguisePage from "../pages/DisguisePage.vue";

const routes = [
  { path: "/", component: DisguiseMenu },
  { path: "/disguise", component: DisguisePage },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
