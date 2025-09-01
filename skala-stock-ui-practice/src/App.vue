<script>
import { defineComponent } from "vue";
import NoticeModal from "@/components/teleports/NoticeModal.vue";
import SpinnerModal from "@/components/teleports/SpinnerModal.vue";
import ToastPopup from "@/components/teleports/ToastPopup.vue";
import { RouterView } from "vue-router";

export default defineComponent({
    components: {
        NoticeModal,
        SpinnerModal,
        ToastPopup,
        RouterView,
    },
});
</script>

<template>
    <NoticeModal />
    <SpinnerModal />
    <ToastPopup />
    <RouterView />
</template>
