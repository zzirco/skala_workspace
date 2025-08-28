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
        disabled
      />
    </div>
    <div class="mb-2">
      <label for="weightInput" class="form-label">체중 (kg)</label>
      <input
        id="weightInput"
        type="number"
        class="form-control"
        v-model="weight"
        disabled
      />
    </div>
    <div class="mt-3">
      <p>BMI 지수: {{ bmi }} {{ bmiMessage }}</p>
    </div>
  </div>
  <HongsDietHabit
    title="🍔 음식 먹기"
    :habits="eatingHabits"
    @weightChanged="addWeight"
  />
  <HongsDietHabit
    title="🏃‍♂️ 기술 연습"
    :habits="traningHabits"
    @weightChanged="addWeight"
  />
</template>
<script setup>
import { ref, computed, watch } from "vue";
import HongsDietHabit from "./HongsDietHabit.vue";

const height = ref(170);
const weight = ref(60);

const eatingHabits = [
  { name: "햄버거 (+1kg)", weight: 1 },
  { name: "피자 (+2kg)", weight: 2 },
];
const traningHabits = [
  { name: "걷기 (-1kg)", weight: -1 },
  { name: "달리기 (-2kg)", weight: -2 },
];

function addWeight(w) {
  weight.value += w;
}

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
</script>
