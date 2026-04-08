close all
clearvars

set(0, 'DefaultFigureVisible', 'off');

run('mainRandomWalk.m');

results = struct();
results.config = struct();
results.config.use_2d = Use2D;
results.config.gamma_vector = gammaVector;
results.config.migrate_proportional_vector = migrateProportionalVector;
results.config.sim_seed_vector = simSeedVector;

results.sim_param_vector = simParamVector;
results.time_value = timeIterativeAvrMatrix;
results.time_policy = timePolicyAvrMatrix;
results.time_th_policy = timeThPolicyAvrMatrix;
results.value_error = valueErrorAvrStore;
results.value_policy = valuePolicyAvrMatrix;
results.value_value_result = valueValueResultAvrStore;
results.value_value_actual = valueValueActualAvrStore;
results.value_never = valueNeverMigrateAvrMatrix;
results.value_always = valueAlwaysMigrateAvrMatrix;
results.value_myopic = valueMyopicAvrMatrix;
results.different_action_pct = differentActionCountPerctStrore;
results.value_th_policy = valuesThPolicyActualAvrMatrix;

jsonText = jsonencode(results, 'PrettyPrint', true);
fid = fopen('matlab_random_walk_results.json', 'w');
fprintf(fid, '%s', jsonText);
fclose(fid);

disp('Saved matlab_random_walk_results.json')
