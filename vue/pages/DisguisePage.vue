<template>
  <h1>둔갑술 시전!</h1>
  <p>
    홍길동이 <strong>{{ target }}</strong
    >(으)로 변신했습니다!
  </p>
</template>
<script setup>
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";

const route = useRoute();
const router = useRouter();
const target = ref("");

onMounted(() => {
  if (route.query.target) {
    target.value = route.query.target;
    router.replace({ path: route.path });
  }
});
</script>
