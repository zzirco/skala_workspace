import { createRouter, createWebHistory } from 'vue-router';
import StartMain from '@/pages/start/StartMain.vue'
import StockMain from '@/pages/stock/StockMain.vue'
import WrongRoutingPage from './components/WrongRoutingPage.vue';

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            name: '/',
            component: StartMain
        },
        {
            path: '/start',
            name: '/start',
            component: StartMain
        },
        {
            path: '/stock',
            name: '/stock',
            component: StockMain,
            children: [],
        },
        {
            path: '/wrong',
            name: '/wrong',
            component: WrongRoutingPage,
        },
        {
            path: "/:pathMatch(.*)*",
            redirect: "/wrong",
        }
    ]
})

export default router