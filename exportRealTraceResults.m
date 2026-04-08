close all
clearvars

set(0, 'DefaultFigureVisible', 'off');

run('mainRealCellLocation.m');

summary = struct( ...
    'never', mean(sum(traceOneTimeslotCostNeverMigrate, 2) ./ totalUsers), ...
    'always', mean(sum(traceOneTimeslotCostAlwaysMigrate, 2) ./ totalUsers), ...
    'myopic', mean(sum(traceOneTimeslotCostMyopic, 2) ./ totalUsers), ...
    'threshold', mean(sum(traceOneTimeslotCostThPolicy, 2) ./ totalUsers));

stdSummary = struct( ...
    'never', std(sum(traceOneTimeslotCostNeverMigrate, 2) ./ totalUsers), ...
    'always', std(sum(traceOneTimeslotCostAlwaysMigrate, 2) ./ totalUsers), ...
    'myopic', std(sum(traceOneTimeslotCostMyopic, 2) ./ totalUsers), ...
    'threshold', std(sum(traceOneTimeslotCostThPolicy, 2) ./ totalUsers));

gainOverNever = sum(traceOneTimeslotCostThPolicy, 2) ./ sum(traceOneTimeslotCostNeverMigrate, 2);
gainOverAlways = sum(traceOneTimeslotCostThPolicy, 2) ./ sum(traceOneTimeslotCostAlwaysMigrate, 2);
gainOverMyopic = sum(traceOneTimeslotCostThPolicy, 2) ./ sum(traceOneTimeslotCostMyopic, 2);

gainOverNever = gainOverNever(~isnan(gainOverNever));
gainOverAlways = gainOverAlways(~isnan(gainOverAlways));
gainOverMyopic = gainOverMyopic(~isnan(gainOverMyopic));

gainStats = struct();
gainStats.gain_over_never = struct('mean', mean(gainOverNever), 'std', std(gainOverNever), 'min', min(gainOverNever), 'max', max(gainOverNever));
gainStats.gain_over_always = struct('mean', mean(gainOverAlways), 'std', std(gainOverAlways), 'min', min(gainOverAlways), 'max', max(gainOverAlways));
gainStats.gain_over_myopic = struct('mean', mean(gainOverMyopic), 'std', std(gainOverMyopic), 'min', min(gainOverMyopic), 'max', max(gainOverMyopic));

firstMigrate = zeros(length(numberOfUsersInCell(:, 1)), 1);
for timeslot = 1:length(numberOfUsersInCell(:, 1))
    tmp = find(actionsThPolicyEachTimeslot(timeslot, :) < [1:numStates2D + 1], 1);
    if isempty(tmp)
        firstMigrate(timeslot) = numStates2D;
    else
        firstMigrate(timeslot) = tmp - 1;
    end
end

firstMigrateStats = struct( ...
    'mean', mean(firstMigrate), ...
    'std', std(firstMigrate), ...
    'min', min(firstMigrate), ...
    'max', max(firstMigrate));

results = struct();
results.config = struct();
results.config.gamma = gamma;
results.config.max_user_each_cloud = maxUserEachCloud;
results.config.num_cells_with_cloud = numCellsWithCloud;
results.config.avail_resource_trans_factor = availResourceTransFactor;
results.config.avail_resource_migration_factor = availResourceMigrationFactor;
results.config.num_states_2d = numStates2D;
results.summary = summary;
results.std_summary = stdSummary;
results.avg_cost_series = struct( ...
    'never', sum(traceOneTimeslotCostNeverMigrate, 2) ./ totalUsers, ...
    'always', sum(traceOneTimeslotCostAlwaysMigrate, 2) ./ totalUsers, ...
    'myopic', sum(traceOneTimeslotCostMyopic, 2) ./ totalUsers, ...
    'threshold', sum(traceOneTimeslotCostThPolicy, 2) ./ totalUsers);
results.gain_stats = gainStats;
results.first_migrate_stats = firstMigrateStats;
results.time_axis = xPointsEpochs;

jsonText = jsonencode(results, 'PrettyPrint', true);
fid = fopen('matlab_real_trace_results.json', 'w');
fprintf(fid, '%s', jsonText);
fclose(fid);

disp('Saved matlab_real_trace_results.json')
