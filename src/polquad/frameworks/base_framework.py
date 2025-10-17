class BaseFramework:

    def __init__(self, config, bias_calculator, client, framework_name):
        self.config = config
        self.client = client
        self.bias_calculator = bias_calculator
        self.frame_work_name = framework_name
        self.max_iters = self.config['max_iters']
        self.bias_threshold = self.config['bias_threshold']
        self.verbose = self.config['verbose']

    def run_analysis(self, index, row, initial_bias_coords, initial_bias_mag):
        """Orchestrates the analysis for a single statement."""
        # Load data
        statement = row['text']
        specific_data = self._get_specific_framework_data(statement)
        statement_history = self._run_moderation_loop(statement, initial_bias_coords, initial_bias_mag, specific_data)

        return self._build_result_dict(index, row, initial_bias_coords, initial_bias_mag, statement_history, specific_data)
    
    def _run_moderation_loop(self, statement, original_bias, original_mag, specific_data):
        """Handles the iterative moderation process."""
        current_iter = 0
        moderated_mag = original_mag
        converged = False

        history = {0: {"original_statement": statement, "bias": original_bias, "magnitude": original_mag}}

        while moderated_mag > self.bias_threshold and current_iter < self.max_iters:
            if current_iter == 0 and "opinions" in specific_data and self.verbose:
                print("  ↳ Generating Expert Opinions...")


            moderated_statement = self._get_moderated_statement(history, self.bias_threshold, specific_data, is_first_run=(current_iter == 0))
            moderated_bias, moderated_mag = self.bias_calculator.calculate_bias(moderated_statement)

            if self.verbose:
                print(f"  Starting Moderation Attempt {current_iter + 1}/{self.max_iters}")
                print(f"  Moderated Statement: '{moderated_statement}'")
                print(f"  Moderated Bias Coordinates: ({moderated_bias['x']}, {moderated_bias['y']})")
                print(f"  Moderated Magnitude: {moderated_mag:.2f}")

            if moderated_mag <= self.bias_threshold:
                converged = True
            
            history[current_iter + 1] = {
                "moderated_statement": moderated_statement,
                "bias": moderated_bias,
                "magnitude": moderated_mag,
                "converged": converged
            }
            current_iter += 1

        return history
    
    def _get_moderated_statement(self, history, is_first_run):
        raise NotImplementedError("Hey, Dummy... You forgot moderation logic!!")
    
    def _build_result_dict(self, index, row, original_bias, original_mag, history, specific_data):
        """Builds the final JSON output."""
        num_iterations = len(history) - 1
        final_entry = history.get(num_iterations, history[0])
        final_mag = final_entry.get('magnitude', original_mag)

        if original_mag > 0:
            bias_reduction = 100 * (original_mag - final_mag) / original_mag
        else:
            bias_reduction = 0

        framework_data = {
            "moderation_history": history,
                "final_bias_magnitude": final_mag,
                "num_iterations": num_iterations,
                "converged": final_entry.get('converged', False),
                "bias_reduction": bias_reduction
        }
        framework_data.update(specific_data)

        return {
            "index": index,
            "original_statement": row['text'],
            "true_label": row['quadrant'] if 'quadrant' in row else None,
            "initial_bias_coords": original_bias,
            "initial_bias_magnitude": original_mag,
            "frameworks": {
                self.frame_work_name: framework_data
            }
        }
    
    def _get_specific_framework_data(self, statement):
        """Optional method for child classes to add extra data"""
        return {}
