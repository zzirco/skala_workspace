<template>
  <div class="m-2">
    <div v-for="(option, index) in options" class="form-check" :key="index">
      <input
        v-model="inputValue"
        class="form-check-input"
        type="radio"
        :value="option.value"
        :name="elementId"
        :id="elementId + index"
        @click="emitChange(option)"
      />
      <label class="form-check-label" :for="elementId + index">
        {{ option.label }}
      </label>
    </div>
  </div>
</template>
<script setup>
const props = defineProps({
  options: Array,
});

const emit = defineEmits(["changed"]);

const elementId = crypto.randomUUID();

const inputValue = defineModel();

function emitChange(option) {
  emit("changed", option);
}
</script>
