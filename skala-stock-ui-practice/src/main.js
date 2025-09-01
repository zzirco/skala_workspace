import { createApp, defineAsyncComponent } from 'vue';

import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap/dist/js/bootstrap.bundle';
import 'bootstrap-icons/font/bootstrap-icons.css';

import router from './router';
import App from './App.vue'

const app = createApp(App);
app.use(router);

const COMPONENTS = [
  "InlineInput", "InlineTextarea", "OptionSelect", "OptionSwitch",
  "OptionRadio", "TooltipBox", "ItemsTable", "InlineCheckbox", "PageNavigator",
];
COMPONENTS.forEach((component) => {
  app.component(component, defineAsyncComponent(() => import(`./components/${component}.vue`)));
})

app.mount('#app')
