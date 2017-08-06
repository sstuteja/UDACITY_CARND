function out = AstarHeuristic(rows, columns)

out = zeros(rows, columns);
count = 0;
col = size(out, 2);
while true
  startindex = [find(out(:, col) == 0, 1, 'last'), col];
  out(startindex(1), startindex(2)) = count;
  while startindex(1)-1 <= size(out, 1) && startindex(1)-1 >= 1 && startindex(2)+1 <= size(out, 2) && startindex(2)+1 >= 1
    startindex = [startindex(1)-1, startindex(2)+1];
    out(startindex(1), startindex(2)) = count;
  end
  count = count + 1;
  col = max(col - 1, 1);
  if isempty(find(out(:, col) == 0))
    break;
  end
end

end