<template>
  <div class="m-3">
    <h3>홍길동의 BMI 계산기</h3>
    <div class="mb-2">
      <label for="heightInput" class="form-label">키 (cm)</label>
      <input
        id="heightInput"
        type="number"
        class="form-control"
        v-model="height"
      />
    </div>
    <div class="mb-2">
      <label for="weightInput" class="form-label">체중 (kg)</label>
      <input
        id="weightInput"
        type="number"
        class="form-control"
        v-model="weight"
      />
    </div>
    <div class="mt-3">
      <p>BMI 지수: {{ bmi }}</p>
      <p>판정: {{ bmiMessage }}</p>
    </div>
  </div>
</template>
<script setup>
import { ref, computed, watch, onMounted, onBeforeMount } from "vue";

const height = ref(170);
const weight = ref(60);

const bmi = computed(() => {
  console.log(weight.value, height.value);
  return (weight.value / (height.value / 100) ** 2).toFixed(2);
});
const bmiMessage = ref("");

watch(
  [height, weight],
  () => {
    if (bmi.value < 18.5) {
      bmiMessage.value = "저체중";
    } else if (bmi.value < 22.9) {
      bmiMessage.value = "정상";
    } else if (bmi.value < 24.9) {
      bmiMessage.value = "과체중";
    } else {
      bmiMessage.value = "비만";
    }
  },
  { immediate: true }
);

onMounted(() => {
  console.log("on Mounted");
});

onBeforeMount(() => {
  console.log("on Unmounted");
});
</script>
