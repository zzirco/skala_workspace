<template>
  <h1>{{ title }}</h1>
  <div>
    <button v-for="habit in habits" @click="doHabit(habit.weight)">
      {{ habit.name }}
    </button>
  </div>
</template>

<script setup>
const props = defineProps({
  title: String,
  habits: Array,
});

const emit = defineEmits(["weightChanged"]);

function doHabit(w) {
  console.log(w);
  emit("weightChanged", w);
}
</script>
